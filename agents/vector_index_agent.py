# agents/vector_index_agent.py
"""
Agent 2: Vector Index Builder - Modified for Airgap Environment
Creates and manages vector embeddings for code chunks using FAISS and ChromaDB
Uses local CodeBERT model with custom embedding function for ChromaDB
"""

import asyncio
import sqlite3
import json
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import pickle
import hashlib
from datetime import datetime as dt
import uuid
import torch
from transformers import AutoTokenizer, AutoModel
#from typing import Dict, List, Optional, Any, Tuple, Union
import faiss
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
from vllm import AsyncLLMEngine, SamplingParams
from agents.base_agent import BaseOpulenceAgent
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from .code_parser_agent import CodeChunk
import os
# Disable external connections for airgap environment
os.environ['DISABLE_TELEMETRY'] = '1'
os.environ['NO_PROXY'] = '*'
def _ensure_airgap_environment(self):
    """Ensure no external connections are possible"""
    import os
    
    # Set environment variables to disable external connections
    os.environ.update({
        'NO_PROXY': '*',
        'DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        'REQUESTS_CA_BUNDLE': '',
        'CURL_CA_BUNDLE': ''
    })
    
    # Disable requests library
    try:
        import requests
        def blocked_request(*args, **kwargs):
            raise requests.exceptions.ConnectionError("External connections disabled")
        
        requests.get = blocked_request
        requests.post = blocked_request
        requests.request = blocked_request
    except ImportError:
        pass
    
from dataclasses import dataclass

@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    program_name: str
    chunk_id: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int

class LocalCodeBERTEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using local CodeBERT model for ChromaDB"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = "cpu"):
        """
        Initialize with local model paths - ALWAYS USE CPU
        
        Args:
            model_path: Path to local CodeBERT model directory
            tokenizer_path: Path to tokenizer (if different from model_path)
            device: Device to run model on - FORCED TO CPU
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = "cpu"  # ✅ FORCE CPU ALWAYS
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the local CodeBERT model and tokenizer - CPU ONLY"""
        try:
            # Load from local directory - CPU ONLY
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True  # Prevent any external calls
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True  # Prevent any external calls
            )
            self.model.to("cpu")  # ✅ FORCE CPU
            self.model.eval()
            print(f"✅ Loaded local CodeBERT model on CPU (avoiding GPU conflicts)")
        except Exception as e:
            raise RuntimeError(f"Failed to load local CodeBERT model: {e}")
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents - CPU ONLY"""
        try:
            embeddings = []
            
            for text in input:
                # Tokenize - CPU ONLY
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )  # ✅ NO .to(device) - stays on CPU
                
                # Generate embedding - CPU ONLY
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Already on CPU
                    
                    # Normalize for cosine similarity
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding.flatten().tolist())
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")
        
class VectorIndexAgent(BaseOpulenceAgent):  # ✅ INHERIT FROM BASE CLASS
    """Agent for building and managing vector indices - with automatic resource management"""
    
    def __init__(self, coordinator, llm_engine: AsyncLLMEngine = None, 
                 db_path: str = "opulence_data.db", gpu_id: int = 0, 
                 local_model_path: str = None):
        
        # ✅ CALL BASE CLASS CONSTRUCTOR FIRST
        super().__init__(coordinator, "vector_index", db_path, gpu_id)
 
        
        # Local model configuration
        self.local_model_path = local_model_path or "./models/microsoft-codebert-base"
        self.embedding_model_name = "microsoft/codebert-base"  # For reference only
        self.tokenizer = None
        self.embedding_model = None
        self.vector_dim = 768
        
        # Custom embedding function for ChromaDB
        self.chroma_embedding_function = None
        
        # FAISS index
        self.faiss_index = None
        self.faiss_index_path = "opulence_faiss.index"
        
        # ChromaDB client with custom embedding function
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "opulence_code_chunks"
        self.collection = None
        
        # Initialize components
        asyncio.create_task(self._initialize_components())


    async def _ensure_vector_initialized(self):
        """Ensure vector components are initialized before use"""
        if not self._vector_initialized:
            await self._initialize_components()
            self._vector_initialized = True

    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM using base class engine management"""
        try:
            # Use current engine from context (set by base class context manager)
            engine = self._engine
            if engine is None:
                raise RuntimeError("No LLM engine available in current context")

            await asyncio.sleep(0.1)

            # Try new API first (with request_id)
            request_id = str(uuid.uuid4())
            try:
                async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                    result = output
                    break
                return result.outputs[0].text.strip()
            except TypeError as e:
                if "request_id" in str(e):
                    # Fallback to old API (without request_id)
                    async for output in engine.generate(prompt, sampling_params):
                        result = output
                        break
                    return result.outputs[0].text.strip()
                else:
                    raise e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""
    
    def safe_json_loads(json_str):
        """Safely load JSON string with fallback"""
        if not json_str:
            return {}
        try:
            if isinstance(json_str, dict):
                return json_str
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    async def _initialize_components(self):
        """Initialize embedding model and vector databases - CPU ONLY FOR CODEBERT"""
        try:
            # Wait a bit to allow coordinator initialization
            await asyncio.sleep(1)
            
            # ✅ FORCE CPU FOR EMBEDDING MODEL TO AVOID GPU CONFLICTS
            embedding_device = "cpu"
            self.logger.info(f"🔧 Using CPU for CodeBERT to avoid GPU conflicts with LLM (LLM uses GPU {self.gpu_id})")
            
            # Validate local model path exists
            if not Path(self.local_model_path).exists():
                raise FileNotFoundError(f"Local model not found at: {self.local_model_path}")
            
            # Load embedding model from local directory - CPU ONLY
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                local_files_only=True  # Prevent any external calls
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.local_model_path,
                local_files_only=True  # Prevent any external calls
            )
            self.embedding_model.to(embedding_device)  # ✅ CPU ONLY
            self.embedding_model.eval()
            
            self.logger.info(f"✅ Loaded local CodeBERT model on {embedding_device}")
            
            # Create custom embedding function for ChromaDB - CPU ONLY
            self.chroma_embedding_function = LocalCodeBERTEmbeddingFunction(
                model_path=self.local_model_path,
                device=embedding_device  # ✅ CPU ONLY
            )
            
            # Initialize FAISS index
            if Path(self.faiss_index_path).exists():
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                self.logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
                self.logger.info("Created new FAISS index")
            
            # Initialize ChromaDB collection with custom embedding function
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.chroma_embedding_function  # Use custom function
                )
                self.logger.info(f"Loaded existing ChromaDB collection with local embeddings")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.chroma_embedding_function,  # Use custom function
                    metadata={"description": "Opulence mainframe code chunks - local embeddings"}
                )
                self.logger.info("Created new ChromaDB collection with local embeddings")
            
            self._vector_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector components: {str(e)}")
            raise
    
    async def embed_code_chunk(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for a code chunk using local model - CPU ONLY"""
        try:
            # Prepare text for embedding
            text_to_embed = self._prepare_text_for_embedding(chunk_content, chunk_metadata)
            
            # ✅ FORCE CPU FOR TOKENIZATION - NO GPU USAGE
            embedding_device = "cpu"
            inputs = self.tokenizer(
                text_to_embed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )  # ✅ NO .to(device) - stays on CPU
            
            # Generate embedding - CPU ONLY
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Already on CPU
                
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.flatten()
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            # Return zero embedding as fallback
            return np.zeros(self.vector_dim)
    
    def _prepare_text_for_embedding(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding by combining content and metadata"""
        # Start with the actual code content
        text_parts = [content.strip()]
        
        # Add metadata as context safely
        if metadata and isinstance(metadata, dict):
            if 'main_purpose' in metadata:
                text_parts.append(f"Purpose: {metadata['main_purpose']}")
            
            if 'field_names' in metadata and isinstance(metadata['field_names'], list):
                text_parts.append(f"Fields: {', '.join(str(f) for f in metadata['field_names'][:10])}")
            
            if 'operations' in metadata and isinstance(metadata['operations'], list):
                text_parts.append(f"Operations: {', '.join(str(op) for op in metadata['operations'][:5])}")
            
            if 'file_operations' in metadata and isinstance(metadata['file_operations'], list):
                text_parts.append(f"File ops: {', '.join(str(op) for op in metadata['file_operations'][:5])}")
        
        return " | ".join(text_parts)

    async def create_embeddings_for_chunks(self, chunks: List[Union[tuple, 'CodeChunk']]) -> Dict[str, Any]:
        """Create embeddings for a list of chunks with automatic engine management"""
        
        # ✅ USE BASE CLASS CONTEXT MANAGER
        async with self.get_engine_context() as engine:
            try:
                await self._ensure_vector_initialized()
                
                embeddings_created = 0
                
                for chunk_data in chunks:
                    try:
                        if isinstance(chunk_data, tuple):
                            chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk_data
                            metadata = json.loads(metadata_str) if metadata_str else {}
                        else:  # It's a CodeChunk object
                            chunk_id = getattr(chunk_data, 'chunk_id', str(uuid.uuid4()))
                            program_name = chunk_data.program_name
                            chunk_id_str = chunk_data.chunk_id
                            chunk_type = chunk_data.chunk_type
                            content = chunk_data.content
                            metadata = chunk_data.metadata if chunk_data.metadata else {}
                        
                        # Ensure metadata is a dictionary
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        # Generate embedding using local model
                        embedding = await self.embed_code_chunk(content, metadata)
                        
                        # Store in FAISS
                        faiss_id = self.faiss_index.ntotal
                        self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
                        
                        # Store in ChromaDB (will use local embedding function automatically)
                        chroma_metadata = {
                            "program_name": program_name,
                            "chunk_id": chunk_id_str,
                            "chunk_type": chunk_type,
                            "faiss_id": faiss_id
                        }
                        
                        # Add other metadata safely
                        for key, value in metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                chroma_metadata[key] = value
                            elif isinstance(value, list):
                                chroma_metadata[key] = json.dumps(value)
                            elif isinstance(value, dict):
                                chroma_metadata[key] = json.dumps(value)
                            else:
                                chroma_metadata[key] = str(value)
                        
                        self.collection.add(
                            documents=[content],
                            metadatas=[chroma_metadata],
                            ids=[f"{program_name}_{chunk_id_str}"]
                        )
                        
                        # Store embedding reference in SQLite
                        embedding_id = f"{program_name}_{chunk_id_str}_embed"
                        await self._store_embedding_reference(
                            chunk_id if isinstance(chunk_id, int) else hash(chunk_id_str), 
                            embedding_id, 
                            faiss_id, 
                            embedding.tolist()
                        )
                        
                        embeddings_created += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process chunk {chunk_id_str}: {str(e)}")
                        continue
                
                         
                # Final save
                await self._save_faiss_index()
                
                result = {
                    "status": "success",
                    "total_chunks": len(chunks),
                    "embeddings_created": embeddings_created,
                    "faiss_index_size": self.faiss_index.ntotal,
                    'gpu_used': self.gpu_id,
                    'agent_type': self.agent_type
                }
                
                return result
                
            except Exception as e:
                    self.logger.error(f"Batch embedding processing failed: {str(e)}")
                    return {"status": "error", "error": str(e)}

    async def search_similar_components(self, component_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for components similar to the given component name"""
        try:
            await self._ensure_vector_initialized()
            
            # Create search query based on component name
            search_query = f"component {component_name} similar functionality"
            
            # Perform semantic search
            results = await self.semantic_search(search_query, top_k)
            
            # Filter and enhance results
            similar_components = []
            for result in results:
                metadata = result.get('metadata', {})
                
                # Skip exact matches
                if metadata.get('program_name') == component_name:
                    continue
                    
                similar_components.append({
                    "component_name": metadata.get('program_name', 'Unknown'),
                    "chunk_id": metadata.get('chunk_id', 'Unknown'),
                    "chunk_type": metadata.get('chunk_type', 'Unknown'),
                    "similarity_score": result.get('similarity_score', 0),
                    "content_preview": result.get('content', '')[:200] + "...",
                    "shared_elements": self._find_shared_elements(component_name, metadata)
                })
            
            search_result = {
                "status": "success",
                "component_name": component_name,
                "similar_components": similar_components,
                "total_found": len(similar_components)
            }
            
            return self._add_processing_info(search_result)
            
        except Exception as e:
            self.logger.error(f"Similar component search failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

    async def rebuild_index_from_chunks(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Rebuild index from provided chunks"""
        try:
            await self._ensure_vector_initialized()
            
            # Clear existing indices
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Clear ChromaDB collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.chroma_embedding_function,  # Use local function
                metadata={"description": "Opulence mainframe code chunks - rebuilt"}
            )
            
            # Process all chunks
            result = await self.create_embeddings_for_chunks(chunks)
            
            rebuild_result = {
                "status": "success",
                "message": "Index rebuilt from chunks",
                "chunks_processed": len(chunks),
                **result
            }
            
            return self._add_processing_info(rebuild_result)
            
        except Exception as e:
            self.logger.error(f"Index rebuild from chunks failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

    def _find_shared_elements(self, component_name: str, metadata: Dict[str, Any]) -> List[str]:
        """Find shared elements between components"""
        shared = []
        
        # This is a simple implementation - you could enhance it
        if 'field_names' in metadata:
            shared.append(f"Fields: {', '.join(metadata['field_names'][:3])}")
        
        if 'operations' in metadata:
            shared.append(f"Operations: {', '.join(metadata['operations'][:3])}")
        
        if 'file_operations' in metadata:
            shared.append(f"File ops: {', '.join(metadata['file_operations'][:2])}")
        
        return shared
    
    async def process_batch_embeddings(self, limit: int = None) -> Dict[str, Any]:
        """Process all unembedded chunks in batch"""
        try:
            await self._ensure_vector_initialized()
            # Get chunks that need embedding
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT id, program_name, chunk_id, chunk_type, content, metadata, embedding_id
                FROM program_chunks 
                WHERE embedding_id NOT IN (
                    SELECT DISTINCT embedding_id FROM vector_embeddings WHERE embedding_id IS NOT NULL
                )
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return self._add_processing_info({"status": "no_chunks_to_process", "processed": 0})
            
            embeddings_created = 0
            batch_size = 10  # Process in batches to avoid memory issues
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Process batch
                for chunk_data in batch:
                    chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str, embedding_id = chunk_data
                    
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else {}
                        
                        # Generate embedding using local model
                        embedding = await self.embed_code_chunk(content, metadata)
                        
                        # Store in FAISS
                        faiss_id = self.faiss_index.ntotal
                        self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
                        
                        # Store in ChromaDB (uses local embedding function)
                        self.collection.add(
                            documents=[content],
                            metadatas=[{
                                "program_name": program_name,
                                "chunk_id": chunk_id_str,
                                "chunk_type": chunk_type,
                                "faiss_id": faiss_id,
                                **metadata
                            }],
                            ids=[f"{program_name}_{chunk_id_str}"]
                        )
                        
                        # Store embedding reference in SQLite
                        await self._store_embedding_reference(
                            chunk_id, embedding_id, faiss_id, embedding.tolist()
                        )
                        
                        embeddings_created += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process chunk {chunk_id_str}: {str(e)}")
                        continue
                
                # Save FAISS index periodically
                if i % (batch_size * 5) == 0:
                    await self._save_faiss_index()
            
            # Final save
            await self._save_faiss_index()
            
            result = {
                "status": "success",
                "total_chunks": len(chunks),
                "embeddings_created": embeddings_created,
                "faiss_index_size": self.faiss_index.ntotal
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Batch embedding processing failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})
        
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'vector_index'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result
        

    async def _store_embedding_reference(self, chunk_id: int, embedding_id: str, 
                                       faiss_id: int, embedding_vector: List[float]):
        """Store embedding reference in SQLite"""
        await self._ensure_vector_initialized()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                embedding_id TEXT,
                faiss_id INTEGER,
                embedding_vector TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES program_chunks (id)
            )
        """)
        
        cursor.execute("""
            INSERT OR REPLACE INTO vector_embeddings 
            (chunk_id, embedding_id, faiss_id, embedding_vector)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, embedding_id, faiss_id, json.dumps(embedding_vector)))
        
        conn.commit()
        conn.close()
    
    async def _save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            self.logger.info(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {str(e)}")
    
    async def semantic_search(self, query: str, top_k: int = 10, 
                        filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            await self._ensure_vector_initialized()
            
            if self.faiss_index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = await self.embed_code_chunk(query, {})
            
            # Safe search with bounds checking
            search_k = min(top_k * 2, self.faiss_index.ntotal)
            if search_k <= 0:
                return []
                
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                search_k
            )
            
            results = []
            # Safe iteration with bounds checking
            if len(scores) > 0 and len(indices) > 0:
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or idx < 0:
                        continue
                    
                    try:
                        # Safe ChromaDB query
                        chroma_results = self.collection.query(
                            query_texts=[query],
                            where={"faiss_id": int(idx)},
                            n_results=1
                        )
                        
                        # Safe result extraction
                        if (chroma_results and 
                            'documents' in chroma_results and 
                            len(chroma_results['documents']) > 0 and
                            len(chroma_results['documents'][0]) > 0):
                            
                            # Safe metadata extraction
                            metadata = {}
                            if ('metadatas' in chroma_results and 
                                len(chroma_results['metadatas']) > 0 and
                                len(chroma_results['metadatas'][0]) > 0):
                                metadata = chroma_results['metadatas'][0][0]
                            
                            results.append({
                                "content": chroma_results['documents'][0][0],
                                "metadata": metadata,
                                "similarity_score": float(score),
                                "faiss_id": int(idx)
                            })
                            
                    except Exception as e:
                        self.logger.error(f"Error retrieving result {idx}: {str(e)}")
                        continue
            
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def find_similar_code_patterns(self, reference_chunk_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Find similar code patterns to a reference chunk"""
            try:
                await self._ensure_vector_initialized()
                # Get reference chunk embedding
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ve.embedding_vector, pc.content, pc.metadata
                    FROM vector_embeddings ve
                    JOIN program_chunks pc ON ve.chunk_id = pc.id
                    WHERE pc.chunk_id = ?
                """, (reference_chunk_id,))
                
                result = cursor.fetchone()
                conn.close()
                
                if not result:
                    return []
                
                ref_embedding = np.array(json.loads(result[0]))
                ref_content = result[1]
                ref_metadata = json.loads(result[2]) if result[2] else {}
                
                # Search for similar patterns
                scores, indices = self.faiss_index.search(
                    ref_embedding.reshape(1, -1).astype('float32'), 
                    top_k + 1  # +1 because reference will be included
                )
                
                similar_patterns = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    
                    # Get chunk data using local embeddings
                    chroma_results = self.collection.query(
                        query_texts=[ref_content],  # Use local embedding function
                        where={"faiss_id": int(idx)},
                        n_results=1
                    )
                    
                    if chroma_results['documents']:
                        metadata = chroma_results['metadatas'][0][0]
                        
                        # Skip if this is the reference chunk itself
                        if metadata.get('chunk_id') == reference_chunk_id:
                            continue
                        
                        similar_patterns.append({
                            "content": chroma_results['documents'][0][0],
                            "metadata": metadata,
                            "similarity_score": float(score),
                            "pattern_type": await self._analyze_pattern_similarity(
                                ref_content, chroma_results['documents'][0][0]
                            )
                        })
                
                return similar_patterns[:top_k]
                
            except Exception as e:
                self.logger.error(f"Pattern search failed: {str(e)}")
                return []
    
    async def _analyze_pattern_similarity(self, ref_code: str, similar_code: str) -> str:
        """Analyze what makes two code patterns similar"""
        
        await self._ensure_vector_initialized()
        
        prompt = f"""
        Compare these two code patterns and identify the similarity type:
        
        Reference Code:
        {ref_code[:500]}
        
        Similar Code:
        {similar_code[:500]}
        
        What type of similarity do they share?
        Options: structural, functional, data_access, business_logic, error_handling, file_operations
        
        Return only the similarity type.
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50)
        
        try:
            result = await self._generate_with_llm(prompt, sampling_params)
            return result
        except Exception as e:
            self.logger.error(f"Pattern similarity analysis failed: {str(e)}")
            return "structural"  # Default fallback

    async def _enhance_search_query(self, pattern_description: str) -> str:
        """Use LLM to enhance search query"""
        
        await self._ensure_vector_initialized()
        
        prompt = f"""
        Convert this natural language pattern description into a technical search query for mainframe COBOL/JCL code:
        
        Pattern: "{pattern_description}"
        
        Generate a technical query that includes:
        1. Relevant COBOL/JCL keywords
        2. Common field names or operations
        3. Technical terms that would appear in the code
        
        Return only the enhanced query text.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=200)
        
        try:
            result = await self._generate_with_llm(prompt, sampling_params)
            return result
        except Exception as e:
            self.logger.error(f"Query enhancement failed: {str(e)}")
            return pattern_description  # Fallback to original

    async def _rank_search_results(self, original_query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to rank search results by relevance"""
        
        if not results:
            return results
        
        # For each result, get relevance score
        for result in results:
            relevance_score = await self._calculate_relevance_score(
                original_query, result['content'], result['metadata']
            )
            result['relevance_score'] = relevance_score
        
        # Sort by combined similarity and relevance
        def combined_score(r):
            return 0.6 * r['similarity_score'] + 0.4 * r.get('relevance_score', 0)
        
        results.sort(key=combined_score, reverse=True)
        return results
    
    async def _calculate_relevance_score(self, query: str, content: str, metadata: Dict) -> float:
        """Calculate relevance score using LLM"""
        await self._ensure_vector_initialized()
        #await self._ensure_llm_engine()
        
        prompt = f"""
        Rate the relevance of this code chunk to the query on a scale of 0.0 to 1.0:
        
        Query: "{query}"
        
        Code: {content[:300]}...
        
        Metadata: {json.dumps(metadata, indent=2)[:200]}...
        
        Consider:
        1. How well the code matches the query intent
        2. Whether the functionality aligns with what's requested
        3. Code quality and completeness
        
        Return only a decimal number between 0.0 and 1.0.
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50)
        
        try:
            score_text = await self._generate_with_llm(prompt, sampling_params)
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.error(f"Relevance calculation failed: {str(e)}")
            return 0.5  # Default score

    async def build_code_knowledge_graph(self) -> Dict[str, Any]:
        """Build a knowledge graph of code relationships"""
        try:
            await self._ensure_vector_initialized()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all chunks with embeddings
            cursor.execute("""
                SELECT pc.program_name, pc.chunk_id, pc.chunk_type, pc.metadata, ve.faiss_id
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
            """)
            
            chunks = cursor.fetchall()
            conn.close()
            
            knowledge_graph = {
                "nodes": [],
                "edges": [],
                "clusters": {}
            }
            
            # Create nodes
            for program_name, chunk_id, chunk_type, metadata_str, faiss_id in chunks:
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                node = {
                    "id": chunk_id,
                    "program": program_name,
                    "type": chunk_type,
                    "metadata": metadata,
                    "faiss_id": faiss_id
                }
                knowledge_graph["nodes"].append(node)
            
            # Find relationships between chunks
            for i, chunk1 in enumerate(chunks):
                chunk1_id = chunk1[1]
                chunk1_metadata = json.loads(chunk1[3]) if chunk1[3] else {}
                
                for j, chunk2 in enumerate(chunks[i+1:], i+1):
                    chunk2_id = chunk2[1]
                    chunk2_metadata = json.loads(chunk2[3]) if chunk2[3] else {}
                    
                    # Check for relationships
                    relationship = self._find_chunk_relationship(chunk1_metadata, chunk2_metadata)
                    
                    if relationship:
                        edge = {
                            "source": chunk1_id,
                            "target": chunk2_id,
                            "relationship": relationship,
                            "weight": self._calculate_relationship_weight(relationship)
                        }
                        knowledge_graph["edges"].append(edge)
            
            # Perform clustering
            knowledge_graph["clusters"] = await self._cluster_similar_chunks()
            
            return knowledge_graph
            
        except Exception as e:
            self.logger.error(f"Knowledge graph building failed: {str(e)}")
            return {"nodes": [], "edges": [], "clusters": {}}
    
    def _find_chunk_relationship(self, metadata1: Dict, metadata2: Dict) -> Optional[str]:
        """Find relationship between two chunks based on metadata"""
        # Check for shared fields
        fields1 = set(metadata1.get('field_names', []))
        fields2 = set(metadata2.get('field_names', []))
        if fields1 & fields2:
            return "shared_fields"
        
        # Check for perform relationships
        called1 = set(metadata1.get('called_paragraphs', []))
        called2 = set(metadata2.get('called_paragraphs', []))
        if called1 & called2:
            return "shared_calls"
        
        # Check for file operations
        files1 = set(metadata1.get('files', []))
        files2 = set(metadata2.get('files', []))
        if files1 & files2:
            return "shared_files"
        
        # Check for SQL operations on same tables
        tables1 = set(metadata1.get('tables', []))
        tables2 = set(metadata2.get('tables', []))
        if tables1 & tables2:
            return "shared_tables"
        
        return None
    
    def _calculate_relationship_weight(self, relationship: str) -> float:
        """Calculate weight for relationship type"""
        weights = {
            "shared_fields": 0.8,
            "shared_calls": 0.9,
            "shared_files": 0.7,
            "shared_tables": 0.8
        }
        return weights.get(relationship, 0.5)
    
    async def _cluster_similar_chunks(self) -> Dict[str, List[str]]:
        """Cluster similar chunks using embeddings"""
        try:
            if self.faiss_index.ntotal == 0:
                return {}
            
            # Get all embeddings
            all_embeddings = []
            chunk_ids = []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pc.chunk_id, ve.embedding_vector
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
            """)
            
            for chunk_id, embedding_str in cursor.fetchall():
                embedding = np.array(json.loads(embedding_str))
                all_embeddings.append(embedding)
                chunk_ids.append(chunk_id)
            
            conn.close()
            
            if not all_embeddings:
                return {}
            
            # Simple clustering using similarity threshold
            embeddings_matrix = np.vstack(all_embeddings)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
            
            # Form clusters based on similarity threshold
            clusters = {}
            visited = set()
            cluster_id = 0
            
            for i, chunk_id in enumerate(chunk_ids):
                if chunk_id in visited:
                    continue
                
                # Find similar chunks
                similar_indices = np.where(similarity_matrix[i] > 0.7)[0]
                cluster_chunks = [chunk_ids[idx] for idx in similar_indices]
                
                if len(cluster_chunks) > 1:
                    clusters[f"cluster_{cluster_id}"] = cluster_chunks
                    visited.update(cluster_chunks)
                    cluster_id += 1
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            return {}
    
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding index"""
        try:
            await self._ensure_vector_initialized()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count embeddings by type
            cursor.execute("""
                SELECT pc.chunk_type, COUNT(*) as count
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                GROUP BY pc.chunk_type
            """)
            
            type_counts = dict(cursor.fetchall())
            
            # Count embeddings by program
            cursor.execute("""
                SELECT pc.program_name, COUNT(*) as count
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                GROUP BY pc.program_name
                ORDER BY count DESC
                LIMIT 10
            """)
            
            program_counts = dict(cursor.fetchall())
            
            conn.close()
            
            # ChromaDB collection stats
            collection_count = self.collection.count()
            
            result = {
                "total_embeddings": self.faiss_index.ntotal if self.faiss_index else 0,
                "embeddings_by_type": type_counts,
                "top_programs": program_counts,
                "chroma_collection_count": collection_count,
                "vector_dimension": self.vector_dim,
                "index_file_exists": Path(self.faiss_index_path).exists(),
                "local_model_path": self.local_model_path,
                "airgap_compatible": True,
                'gpu_used': self.gpu_id,
                'agent_type': self.agent_type
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding statistics: {str(e)}")
            return {'gpu_used': self.gpu_id, 'agent_type': self.agent_type}
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the entire vector index from scratch"""
        try:
            await self._ensure_vector_initialized()
            # Clear existing indices
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Clear ChromaDB collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.chroma_embedding_function,  # Use local function
                metadata={"description": "Opulence mainframe code chunks - rebuilt"}
            )
            
            # Clear embedding references
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vector_embeddings")
            conn.commit()
            conn.close()
            
            # Rebuild embeddings
            result = await self.process_batch_embeddings()
            
            # Save index
            await self._save_faiss_index()
            
            final_result = {
                "status": "success",
                "message": "Index rebuilt successfully",
                **result
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {str(e)}")
            return {"status": "error", "error": str(e)}


    async def advanced_semantic_search(self, query: str, top_k: int = 10, 
                                     use_reranking: bool = True,
                                     enhance_query: bool = True) -> List[Dict[str, Any]]:
        """Advanced semantic search with query enhancement and result reranking"""
        try:
            await self._ensure_vector_initialized()
            
            # Enhance query if requested
            search_query = query
            if enhance_query:
                search_query = await self._enhance_search_query(query)
                self.logger.info(f"Enhanced query: '{query}' -> '{search_query}'")
            
            # Perform initial semantic search
            initial_results = await self.semantic_search(search_query, top_k * 2)
            
            # Rerank results if requested
            if use_reranking and initial_results:
                final_results = await self._rank_search_results(query, initial_results)
            else:
                final_results = initial_results
            
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Advanced semantic search failed: {str(e)}")
            return []

    async def find_code_dependencies(self, program_name: str) -> Dict[str, Any]:
        """Find code dependencies for a given program using embeddings"""
        try:
            await self._ensure_vector_initialized()
            
            # Search for chunks related to this program
            program_chunks = await self.semantic_search(
                f"program {program_name} dependencies imports calls",
                top_k=20,
                filter_metadata={"program_name": program_name}
            )
            
            dependencies = {
                "direct_calls": [],
                "file_dependencies": [],
                "data_dependencies": [],
                "similar_programs": []
            }
            
            for chunk in program_chunks:
                metadata = chunk.get('metadata', {})
                
                # Extract direct calls
                if 'called_paragraphs' in metadata:
                    dependencies["direct_calls"].extend(metadata['called_paragraphs'])
                
                # Extract file dependencies
                if 'files' in metadata:
                    dependencies["file_dependencies"].extend(metadata['files'])
                
                # Extract data dependencies
                if 'field_names' in metadata:
                    dependencies["data_dependencies"].extend(metadata['field_names'])
            
            # Find similar programs
            similar_programs = await self.search_similar_components(program_name, top_k=5)
            dependencies["similar_programs"] = similar_programs.get('similar_components', [])
            
            # Remove duplicates
            for key in ["direct_calls", "file_dependencies", "data_dependencies"]:
                dependencies[key] = list(set(dependencies[key]))
            
            result = {
                "status": "success",
                "program_name": program_name,
                "dependencies": dependencies,
                "total_chunks_analyzed": len(program_chunks)
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

    async def search_by_functionality(self, functionality_description: str, 
                                    top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for code chunks by functionality description"""
        try:
            await self._ensure_vector_initialized()
            
            # Create multiple search variations
            search_queries = [
                functionality_description,
                f"code that {functionality_description}",
                f"function {functionality_description}",
                f"program {functionality_description}"
            ]
            
            all_results = []
            seen_ids = set()
            
            for query in search_queries:
                results = await self.semantic_search(query, top_k=5)
                
                for result in results:
                    chunk_id = result.get('metadata', {}).get('chunk_id')
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append(result)
            
            # Sort by similarity score and return top results
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return all_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Functionality search failed: {str(e)}")
            return []

    async def export_embeddings(self, output_path: str) -> Dict[str, Any]:
        """Export embeddings and metadata to file"""
        try:
            await self._ensure_vector_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all embeddings with metadata
            cursor.execute("""
                SELECT pc.program_name, pc.chunk_id, pc.chunk_type, 
                       pc.content, pc.metadata, ve.embedding_vector
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
            """)
            
            export_data = {
                "embeddings": [],
                "metadata": {
                    "total_count": 0,
                    "vector_dimension": self.vector_dim,
                    "model_path": self.local_model_path,
                    "export_timestamp": dt.now().isoformat()
                }
            }
            
            for row in cursor.fetchall():
                program_name, chunk_id, chunk_type, content, metadata_str, embedding_str = row
                
                export_data["embeddings"].append({
                    "program_name": program_name,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "content": content,
                    "metadata": json.loads(metadata_str) if metadata_str else {},
                    "embedding": json.loads(embedding_str)
                })
            
            export_data["metadata"]["total_count"] = len(export_data["embeddings"])
            
            conn.close()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            result = {
                "status": "success",
                "output_path": output_path,
                "total_exported": export_data["metadata"]["total_count"],
                "file_size_mb": Path(output_path).stat().st_size / (1024 * 1024)
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

    async def import_embeddings(self, input_path: str) -> Dict[str, Any]:
        """Import embeddings from file"""
        try:
            await self._ensure_vector_initialized()
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for embedding_data in import_data["embeddings"]:
                try:
                    # Add to FAISS
                    embedding_vector = np.array(embedding_data["embedding"])
                    faiss_id = self.faiss_index.ntotal
                    self.faiss_index.add(embedding_vector.reshape(1, -1).astype('float32'))
                    
                    # Add to ChromaDB
                    self.collection.add(
                        documents=[embedding_data["content"]],
                        metadatas=[{
                            "program_name": embedding_data["program_name"],
                            "chunk_id": embedding_data["chunk_id"],
                            "chunk_type": embedding_data["chunk_type"],
                            "faiss_id": faiss_id,
                            **embedding_data["metadata"]
                        }],
                        ids=[f"{embedding_data['program_name']}_{embedding_data['chunk_id']}"]
                    )
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to import embedding: {str(e)}")
                    continue
            
            # Save FAISS index
            await self._save_faiss_index()
            
            result = {
                "status": "success",
                "input_path": input_path,
                "total_imported": imported_count,
                "total_in_file": len(import_data["embeddings"])
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Import failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

