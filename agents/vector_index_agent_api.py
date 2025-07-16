# agents/vector_index_agent_api.py
"""
Agent 2: Vector Index Builder - FIXED for API-Based Architecture
Creates and manages vector embeddings for code chunks using FAISS and ChromaDB
Uses local CodeBERT model with custom embedding function for ChromaDB
FIXED: Import errors, initialization issues, and coordinator compatibility
"""

import asyncio
import sqlite3
import json
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import pickle
import hashlib
from datetime import datetime as dt
import uuid

# FIXED: Conditional imports with proper error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    EmbeddingFunction = None
    Embeddings = None
    Documents = None

# FIXED: Import base agent with proper error handling
try:
    from agents.base_agent_api import BaseOpulenceAgent
except ImportError:
    try:
        from .base_agent_api import BaseOpulenceAgent
    except ImportError:
        # Fallback base class
        class BaseOpulenceAgent:
            def __init__(self, coordinator, agent_type, db_path, gpu_id=None):
                self.coordinator = coordinator
                self.agent_type = agent_type
                self.db_path = db_path
                self.gpu_id = gpu_id
                self.logger = logging.getLogger(f"{__name__}.{agent_type}")

# Disable external connections for airgap environment
os.environ.update({
    'DISABLE_TELEMETRY': '1',
    'NO_PROXY': '*',
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
    'REQUESTS_CA_BUNDLE': '',
    'CURL_CA_BUNDLE': ''
})

def _ensure_airgap_environment():
    """Ensure no external connections are possible"""
    try:
        import requests
        def blocked_request(*args, **kwargs):
            raise requests.exceptions.ConnectionError("External connections disabled")
        
        requests.get = blocked_request
        requests.post = blocked_request
        requests.request = blocked_request
    except ImportError:
        pass

# FIXED: Standalone CodeChunk class to avoid circular imports
from dataclasses import dataclass

@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    program_name: str
    chunk_id: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    line_start: int = 0
    line_end: int = 0

class LocalCodeBERTEmbeddingFunction:
    """Custom embedding function using local CodeBERT model for ChromaDB"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = "cpu"):
        """Initialize with local model paths - ALWAYS USE CPU"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Please install torch and transformers.")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = "cpu"  # FORCE CPU ALWAYS
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the local CodeBERT model and tokenizer - CPU ONLY"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model.to("cpu")
            self.model.eval()
            print(f"✅ Loaded local CodeBERT model on CPU")
        except Exception as e:
            raise RuntimeError(f"Failed to load local CodeBERT model: {e}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
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
                )
                
                # Generate embedding - CPU ONLY
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # Normalize for cosine similarity
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding.flatten().tolist())
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

class VectorIndexAgent(BaseOpulenceAgent):
    """FIXED: Agent for building and managing vector indices - API-based architecture"""
    
    def __init__(self, coordinator, llm_engine=None, 
                 db_path: str = "opulence_data.db", gpu_id: int = None, 
                 local_model_path: str = None):
        
        # FIXED: Call base class constructor first
        super().__init__(coordinator, "vector_index", db_path, gpu_id)
        
        # Store coordinator reference for API calls
        self.coordinator = coordinator
        
        # FIXED: Check dependencies and set availability flags
        self.torch_available = TORCH_AVAILABLE
        self.faiss_available = FAISS_AVAILABLE
        self.chromadb_available = CHROMADB_AVAILABLE
        
        # Local model configuration with fallback
        self.local_model_path = local_model_path or "./models/microsoft-codebert-base"
        self.embedding_model_name = "microsoft/codebert-base"
        self.tokenizer = None
        self.embedding_model = None
        self.vector_dim = 768
        
        # Custom embedding function for ChromaDB
        self.chroma_embedding_function = None
        
        # FAISS index with availability check
        self.faiss_index = None
        self.faiss_index_path = "opulence_faiss.index"
        
        # ChromaDB client with availability check
        self.chroma_client = None
        self.collection_name = "opulence_code_chunks"
        self.collection = None
        
        # Initialization flags
        self._vector_initialized = False
        self._initialization_attempted = False
        
        # FIXED: Don't call async initialization in __init__
        # Store for later initialization
        self._needs_initialization = True
        
        self.logger.info(f"✅ VectorIndexAgent created (dependencies: torch={self.torch_available}, faiss={self.faiss_available}, chromadb={self.chromadb_available})")

    async def _ensure_vector_initialized(self):
        """FIXED: Ensure vector components are initialized before use"""
        if not self._vector_initialized and not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                await self._initialize_components()
                self._vector_initialized = True
            except Exception as e:
                self.logger.error(f"Vector initialization failed: {e}")
                self._vector_initialized = False

    async def _generate_with_api(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using API coordinator instead of direct engine access"""
        try:
            if not self.coordinator:
                raise RuntimeError("No API coordinator available")

            params = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
            
            result = await self.coordinator.call_model_api(prompt, params)
            
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip()
            elif isinstance(result, dict) and "response" in result:
                return result["response"].strip()
            else:
                return str(result).strip()
                
        except Exception as e:
            self.logger.error(f"API generation failed: {str(e)}")
            return ""
    
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

    async def _initialize_components(self):
        """FIXED: Initialize embedding model and vector databases with improved ChromaDB handling"""
        try:
            # Check if required dependencies are available
            if not self.torch_available:
                self.logger.warning("⚠️ PyTorch not available - vector indexing disabled")
                return
            
            if not self.faiss_available:
                self.logger.warning("⚠️ FAISS not available - vector search disabled")
                return
                
            if not self.chromadb_available:
                self.logger.warning("⚠️ ChromaDB not available - vector storage limited")
            
            # Wait a bit to allow coordinator initialization
            await asyncio.sleep(1)
            
            # FORCE CPU FOR EMBEDDING MODEL
            embedding_device = "cpu"
            self.logger.info(f"🔧 Using CPU for CodeBERT to avoid conflicts with API servers")
            
            # Validate local model path exists
            if not Path(self.local_model_path).exists():
                self.logger.warning(f"⚠️ Local model not found at: {self.local_model_path}")
                # Create a minimal fallback setup
                self._create_fallback_setup()
                return
            
            # Load embedding model from local directory - CPU ONLY
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                local_files_only=True
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.local_model_path,
                local_files_only=True
            )
            self.embedding_model.to(embedding_device)
            self.embedding_model.eval()
            
            self.logger.info(f"✅ Loaded local CodeBERT model on {embedding_device}")
            
            # Create custom embedding function for ChromaDB - CPU ONLY
            if self.chromadb_available:
                self.chroma_embedding_function = LocalCodeBERTEmbeddingFunction(
                    model_path=self.local_model_path,
                    device=embedding_device
                )
            
            # Initialize FAISS index
            if self.faiss_available:
                if Path(self.faiss_index_path).exists():
                    try:
                        self.faiss_index = faiss.read_index(self.faiss_index_path)
                        self.logger.info(f"✅ Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to load FAISS index: {e}")
                        self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                else:
                    self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                    self.logger.info("✅ Created new FAISS index")
            
            # Initialize ChromaDB collection - IMPROVED LOGIC
            if self.chromadb_available and self.chroma_embedding_function:
                try:
                    self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                    
                    # Try to get existing collection first
                    try:
                        self.collection = self.chroma_client.get_collection(
                            name=self.collection_name,
                            embedding_function=self.chroma_embedding_function
                        )
                        self.logger.info(f"✅ Loaded existing ChromaDB collection: {self.collection_name}")
                    except Exception:
                        # Collection doesn't exist, create it
                        try:
                            self.collection = self.chroma_client.create_collection(
                                name=self.collection_name,
                                embedding_function=self.chroma_embedding_function,
                                metadata={"description": "Opulence mainframe code chunks - local embeddings"}
                            )
                            self.logger.info(f"✅ Created new ChromaDB collection: {self.collection_name}")
                        except Exception as create_error:
                            self.logger.error(f"❌ Failed to create ChromaDB collection: {create_error}")
                            self.collection = None
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ ChromaDB initialization failed: {e}")
                    self.collection = None
            
            self._vector_initialized = True
            self.logger.info("✅ Vector components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector components: {str(e)}")
            self._create_fallback_setup()
            raise

    def _create_fallback_setup(self):
        """Create a minimal fallback setup when full initialization fails"""
        self.logger.info("🔄 Creating fallback vector setup...")
        
        try:
            # Create minimal FAISS index if available
            if self.faiss_available:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                self.logger.info("✅ Created fallback FAISS index")
        except Exception as e:
            self.logger.warning(f"⚠️ Fallback FAISS creation failed: {e}")
        
        # Set fallback initialized flag
        self._vector_initialized = True
    
    async def embed_code_chunk(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> np.ndarray:
        """FIXED: Generate embedding for a code chunk with proper error handling"""
        try:
            if not self.torch_available or not self.tokenizer or not self.embedding_model:
                # Return zero embedding as fallback
                self.logger.warning("⚠️ Embedding model not available, returning zero embedding")
                return np.zeros(self.vector_dim)
            
            # Prepare text for embedding
            text_to_embed = self._prepare_text_for_embedding(chunk_content, chunk_metadata)
            
            # Force CPU for tokenization
            inputs = self.tokenizer(
                text_to_embed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate embedding - CPU ONLY
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.flatten()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embedding: {str(e)}")
            return np.zeros(self.vector_dim)
    
    def _prepare_text_for_embedding(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding by combining content and metadata"""
        text_parts = [content.strip()]
        
        if metadata and isinstance(metadata, dict):
            if 'main_purpose' in metadata:
                text_parts.append(f"Purpose: {metadata['main_purpose']}")
            
            if 'field_names' in metadata and isinstance(metadata['field_names'], list):
                text_parts.append(f"Fields: {', '.join(str(f) for f in metadata['field_names'][:10])}")
            
            if 'operations' in metadata and isinstance(metadata['operations'], list):
                text_parts.append(f"Operations: {', '.join(str(op) for op in metadata['operations'][:5])}")
        
        return " | ".join(text_parts)

    async def create_embeddings_for_chunks(self, chunks: List[Union[tuple, CodeChunk]]) -> Dict[str, Any]:
        """FIXED: Create embeddings for chunks with comprehensive error handling"""
        try:
            await self._ensure_vector_initialized()
            
            if not chunks:
                return {"status": "success", "embeddings_created": 0, "total_chunks": 0}
            
            embeddings_created = 0
            
            for chunk_data in chunks:
                try:
                    # Parse chunk data safely
                    chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = self._parse_chunk_data(chunk_data)
                    
                    if not content:
                        self.logger.warning(f"⚠️ Empty content for chunk {chunk_id_str}, skipping")
                        continue
                    
                    # Parse metadata safely
                    metadata = self.safe_json_loads(metadata_str)
                    
                    # Generate embedding
                    embedding = await self.embed_code_chunk(content, metadata)
                    
                    if embedding is None or len(embedding) == 0:
                        self.logger.error(f"❌ Failed to generate embedding for chunk {chunk_id_str}")
                        continue
                    
                    # Store in available indices
                    await self._store_embedding_safely(
                        chunk_id, program_name, chunk_id_str, chunk_type, 
                        content, metadata, embedding
                    )
                    
                    embeddings_created += 1
                    self.logger.debug(f"✅ Created embedding for {chunk_id_str}")
                    
                except Exception as chunk_error:
                    self.logger.error(f"❌ Failed to process chunk: {str(chunk_error)}")
                    continue
            
            # Save indices if available
            await self._save_indices_safely()
            
            result = {
                "status": "success",
                "total_chunks": len(chunks),
                "embeddings_created": embeddings_created,
                "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
                'coordinator_type': 'api_based',
                'agent_type': self.agent_type
            }
            
            self.logger.info(f"✅ Created {embeddings_created}/{len(chunks)} embeddings")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Batch embedding processing failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _parse_chunk_data(self, chunk_data):
        """FIXED: Parse chunk data with proper error handling"""
        try:
            if isinstance(chunk_data, tuple):
                if len(chunk_data) >= 6:
                    return chunk_data[:6]
                elif len(chunk_data) >= 5:
                    program_name, chunk_id_str, chunk_type, content, metadata_str = chunk_data[:5]
                    chunk_id = hash(chunk_id_str)
                    return chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str
                else:
                    raise ValueError(f"Invalid tuple format: {len(chunk_data)} elements")
            
            elif hasattr(chunk_data, 'program_name'):  # CodeChunk object
                return (
                    getattr(chunk_data, 'id', hash(chunk_data.chunk_id)),
                    chunk_data.program_name,
                    chunk_data.chunk_id,
                    chunk_data.chunk_type,
                    chunk_data.content,
                    json.dumps(chunk_data.metadata) if chunk_data.metadata else "{}"
                )
            
            else:
                raise ValueError(f"Unknown chunk data format: {type(chunk_data)}")
                
        except Exception as e:
            # Return safe defaults
            return (
                hash(str(chunk_data)),
                "unknown_program",
                f"chunk_{uuid.uuid4().hex[:8]}",
                "unknown",
                str(chunk_data),
                "{}"
            )

    async def _store_embedding_safely(self, chunk_id, program_name, chunk_id_str, chunk_type, content, metadata, embedding):
        """FIXED: Store embedding in available indices with error handling"""
        faiss_id = None
        
        # Store in FAISS if available
        if self.faiss_index and self.faiss_available:
            try:
                faiss_id = self.faiss_index.ntotal
                self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
            except Exception as e:
                self.logger.warning(f"⚠️ FAISS storage failed: {e}")
        
        # Store in ChromaDB if available
        if self.collection and self.chromadb_available:
            try:
                chroma_metadata = {
                    "program_name": str(program_name),
                    "chunk_id": str(chunk_id_str),
                    "chunk_type": str(chunk_type),
                    "faiss_id": int(faiss_id) if faiss_id is not None else -1
                }
                
                # Add metadata safely
                for key, value in metadata.items():
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            chroma_metadata[f"meta_{key}"] = value
                        else:
                            chroma_metadata[f"meta_{key}"] = json.dumps(value)
                    except:
                        pass
                
                unique_id = f"{program_name}_{chunk_id_str}_{faiss_id or 'no_faiss'}"
                self.collection.add(
                    documents=[str(content)],
                    metadatas=[chroma_metadata],
                    ids=[unique_id]
                )
            except Exception as e:
                self.logger.warning(f"⚠️ ChromaDB storage failed: {e}")
        
        # Store reference in SQLite
        try:
            embedding_id = f"{program_name}_{chunk_id_str}_embed"
            await self._store_embedding_reference(
                chunk_id if isinstance(chunk_id, int) else hash(str(chunk_id_str)),
                embedding_id,
                faiss_id or -1,
                embedding.tolist()
            )
        except Exception as e:
            self.logger.warning(f"⚠️ SQLite storage failed: {e}")

    async def _save_indices_safely(self):
        """FIXED: Save indices with proper error handling"""
        if self.faiss_index and self.faiss_available:
            try:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                self.logger.info(f"✅ Saved FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save FAISS index: {e}")

    async def _store_embedding_reference(self, chunk_id: int, embedding_id: str, 
                                       faiss_id: int, embedding_vector: List[float]):
        """FIXED: Store embedding reference in SQLite with better error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            """, (
                int(chunk_id), 
                str(embedding_id), 
                int(faiss_id), 
                json.dumps(embedding_vector)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store embedding reference: {str(e)}")

    async def semantic_search(self, query: str, top_k: int = 10, 
                            filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """FIXED: Semantic search with comprehensive error handling"""
        try:
            await self._ensure_vector_initialized()
            
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                self.logger.warning("⚠️ No vectors available for search")
                return []
            
            # Generate query embedding
            query_embedding = await self.embed_code_chunk(query, {})
            if query_embedding is None or len(query_embedding) == 0:
                self.logger.error("❌ Failed to generate query embedding")
                return []
            
            # Perform search
            search_k = min(top_k * 2, self.faiss_index.ntotal)
            if search_k <= 0:
                return []
                
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                search_k
            )
            
            results = []
            if len(scores) > 0 and len(indices) > 0:
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or idx < 0:
                        continue
                    
                    try:
                        # Get document from ChromaDB if available
                        if self.collection and self.chromadb_available:
                            chroma_results = self.collection.query(
                                query_texts=[query],
                                where={"faiss_id": int(idx)},
                                n_results=1
                            )
                            
                            if (chroma_results and 
                                'documents' in chroma_results and 
                                len(chroma_results['documents']) > 0 and
                                len(chroma_results['documents'][0]) > 0):
                                
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
                        else:
                            # Fallback to SQLite lookup
                            doc_info = await self._get_document_by_faiss_id(idx)
                            if doc_info:
                                results.append({
                                    "content": doc_info.get('content', ''),
                                    "metadata": doc_info.get('metadata', {}),
                                    "similarity_score": float(score),
                                    "faiss_id": int(idx)
                                })
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error retrieving result {idx}: {str(e)}")
                        continue
            
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Semantic search failed: {str(e)}")
            return []

    async def _get_document_by_faiss_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """Get document by FAISS ID from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pc.content, pc.metadata, pc.program_name, pc.chunk_id, pc.chunk_type
                FROM program_chunks pc
                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                WHERE ve.faiss_id = ?
                LIMIT 1
            """, (faiss_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'content': row[0],
                    'metadata': self.safe_json_loads(row[1]),
                    'program_name': row[2],
                    'chunk_id': row[3],
                    'chunk_type': row[4]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"❌ SQLite document lookup failed: {e}")
            return None

    # FIXED: Add all required methods with proper implementations

    async def search_similar_components(self, component_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for components similar to the given component name"""
        try:
            search_query = f"component {component_name} similar functionality"
            results = await self.semantic_search(search_query, top_k)
            
            similar_components = []
            for result in results:
                metadata = result.get('metadata', {})
                
                if metadata.get('program_name') == component_name:
                    continue
                    
                similar_components.append({
                    "component_name": metadata.get('program_name', 'Unknown'),
                    "chunk_id": metadata.get('chunk_id', 'Unknown'),
                    "chunk_type": metadata.get('chunk_type', 'Unknown'),
                    "similarity_score": result.get('similarity_score', 0),
                    "content_preview": result.get('content', '')[:200] + "..."
                })
            
            return {
                "status": "success",
                "component_name": component_name,
                "similar_components": similar_components,
                "total_found": len(similar_components)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Similar component search failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def rebuild_index_from_chunks(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Rebuild index from provided chunks - FIXED ChromaDB handling"""
        try:
            await self._ensure_vector_initialized()
            
            self.logger.info(f"🔄 Rebuilding index from {len(chunks)} chunks")
            
            # Clear existing FAISS index
            if self.faiss_available:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                self.logger.info("✅ Cleared FAISS index")
            
            # Clear ChromaDB collection - FIXED LOGIC
            if self.chromadb_available and self.chroma_client:
                await self._safely_recreate_chroma_collection()
            
            # Clear SQLite embedding references
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vector_embeddings")
                conn.commit()
                conn.close()
                self.logger.info("✅ Cleared SQLite embedding references")
            except Exception as e:
                self.logger.warning(f"⚠️ SQLite cleanup failed: {e}")
            
            # Process all chunks
            result = await self.create_embeddings_for_chunks(chunks)
            
            return {
                "status": "success" if result.get("status") == "success" else "error",
                "message": "Index rebuilt from chunks",
                "chunks_processed": len(chunks),
                "embeddings_created": result.get("embeddings_created", 0),
                "faiss_index_size": result.get("faiss_index_size", 0),
                "coordinator_type": "api_based",
                "agent_type": self.agent_type
            }
            
        except Exception as e:
            self.logger.error(f"❌ Index rebuild failed: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "coordinator_type": "api_based",
                "agent_type": self.agent_type
            }

    async def _safely_recreate_chroma_collection(self):
        """Safely delete and recreate ChromaDB collection"""
        try:
            # First, try to delete the existing collection
            try:
                existing_collection = self.chroma_client.get_collection(self.collection_name)
                if existing_collection:
                    self.chroma_client.delete_collection(self.collection_name)
                    self.logger.info(f"✅ Deleted existing ChromaDB collection: {self.collection_name}")
            except Exception as delete_error:
                # Collection might not exist, which is fine
                self.logger.info(f"💡 Collection {self.collection_name} doesn't exist or couldn't be deleted: {delete_error}")
            
            # Wait a moment for the deletion to complete
            await asyncio.sleep(0.5)
            
            # Now create a new collection
            if self.chroma_embedding_function:
                try:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.chroma_embedding_function,
                        metadata={"description": "Opulence mainframe code chunks - rebuilt", "rebuild_timestamp": dt.now().isoformat()}
                    )
                    self.logger.info(f"✅ Created new ChromaDB collection: {self.collection_name}")
                except Exception as create_error:
                    self.logger.error(f"❌ Failed to create ChromaDB collection: {create_error}")
                    # Try to get existing collection instead
                    try:
                        self.collection = self.chroma_client.get_collection(
                            name=self.collection_name,
                            embedding_function=self.chroma_embedding_function
                        )
                        self.logger.info(f"✅ Retrieved existing ChromaDB collection: {self.collection_name}")
                    except Exception as get_error:
                        self.logger.error(f"❌ Failed to get ChromaDB collection: {get_error}")
                        self.collection = None
            else:
                self.logger.warning("⚠️ No embedding function available for ChromaDB collection creation")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to recreate ChromaDB collection: {e}")
            self.collection = None

    async def process_batch_embeddings(self, limit: int = None) -> Dict[str, Any]:
        """Process all unembedded chunks in batch"""
        try:
            await self._ensure_vector_initialized()
            
            # Get chunks that need embedding
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT id, program_name, chunk_id, chunk_type, content, metadata
                FROM program_chunks 
                WHERE content IS NOT NULL AND content != ''
                AND (embedding_id IS NULL OR embedding_id NOT IN (
                    SELECT DISTINCT embedding_id FROM vector_embeddings WHERE embedding_id IS NOT NULL
                ))
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return {"status": "no_chunks_to_process", "processed": 0}
            
            self.logger.info(f"🔄 Processing {len(chunks)} unembedded chunks")
            
            # Format chunks properly
            formatted_chunks = []
            for chunk_data in chunks:
                chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk_data
                metadata_str = metadata_str or "{}"
                formatted_chunk = (chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str)
                formatted_chunks.append(formatted_chunk)
            
            # Process embeddings
            result = await self.create_embeddings_for_chunks(formatted_chunks)
            
            return {
                **result,
                'coordinator_type': 'api_based',
                'agent_type': 'vector_index'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Batch embedding processing failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding index"""
        try:
            await self._ensure_vector_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count embeddings by type
            try:
                cursor.execute("""
                    SELECT pc.chunk_type, COUNT(*) as count
                    FROM program_chunks pc
                    JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                    GROUP BY pc.chunk_type
                """)
                type_counts = dict(cursor.fetchall())
            except:
                type_counts = {}
            
            # Count embeddings by program
            try:
                cursor.execute("""
                    SELECT pc.program_name, COUNT(*) as count
                    FROM program_chunks pc
                    JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                    GROUP BY pc.program_name
                    ORDER BY count DESC
                    LIMIT 10
                """)
                program_counts = dict(cursor.fetchall())
            except:
                program_counts = {}
            
            conn.close()
            
            # ChromaDB collection stats
            collection_count = 0
            if self.collection and self.chromadb_available:
                try:
                    collection_count = self.collection.count()
                except:
                    collection_count = 0
            
            result = {
                "total_embeddings": self.faiss_index.ntotal if self.faiss_index else 0,
                "embeddings_by_type": type_counts,
                "top_programs": program_counts,
                "chroma_collection_count": collection_count,
                "vector_dimension": self.vector_dim,
                "index_file_exists": Path(self.faiss_index_path).exists(),
                "local_model_path": self.local_model_path,
                "dependencies_available": {
                    "torch": self.torch_available,
                    "faiss": self.faiss_available,
                    "chromadb": self.chromadb_available
                },
                "coordinator_type": "api_based",
                "agent_type": self.agent_type
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get embedding statistics: {str(e)}")
            return {
                "error": str(e),
                "coordinator_type": "api_based", 
                "agent_type": self.agent_type
            }

    async def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the entire vector index from scratch - FIXED ChromaDB handling"""
        try:
            await self._ensure_vector_initialized()
            
            # Clear existing FAISS index
            if self.faiss_available:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                self.logger.info("✅ Cleared FAISS index")
            
            # Clear ChromaDB collection - FIXED LOGIC
            if self.chromadb_available and self.chroma_client:
                await self._safely_recreate_chroma_collection()
            
            # Clear embedding references from SQLite
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vector_embeddings")
                conn.commit()
                conn.close()
                self.logger.info("✅ Cleared SQLite embedding references")
            except Exception as e:
                self.logger.warning(f"⚠️ SQLite cleanup failed: {e}")
            
            # Rebuild embeddings
            result = await self.process_batch_embeddings()
            
            # Save index
            await self._save_indices_safely()
            
            return {
                "status": "success",
                "message": "Index rebuilt successfully",
                **result
            }
            
        except Exception as e:
            self.logger.error(f"❌ Index rebuild failed: {str(e)}")
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
            self.logger.error(f"❌ Advanced semantic search failed: {str(e)}")
            return []

    async def _enhance_search_query(self, pattern_description: str) -> str:
        """Use API to enhance search query"""
        try:
            prompt = f"""
            Convert this natural language pattern description into a technical search query for mainframe COBOL/JCL code:
            
            Pattern: "{pattern_description}"
            
            Generate a technical query that includes:
            1. Relevant COBOL/JCL keywords
            2. Common field names or operations
            3. Technical terms that would appear in the code
            
            Return only the enhanced query text.
            """
            
            result = await self._generate_with_api(prompt, max_tokens=200, temperature=0.2)
            return result if result else pattern_description
            
        except Exception as e:
            self.logger.error(f"❌ Query enhancement failed: {str(e)}")
            return pattern_description

    async def _rank_search_results(self, original_query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use API to rank search results by relevance"""
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
        """Calculate relevance score using API"""
        try:
            prompt = f"""
            Rate the relevance of this code chunk to the query on a scale of 0.0 to 1.0:
            
            Query: "{query}"
            
            Code: {content[:300]}...
            
            Consider:
            1. How well the code matches the query intent
            2. Whether the functionality aligns with what's requested
            3. Code quality and completeness
            
            Return only a decimal number between 0.0 and 1.0.
            """
            
            score_text = await self._generate_with_api(prompt, max_tokens=50, temperature=0.1)
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except:
                return 0.5
            
        except Exception as e:
            self.logger.error(f"❌ Relevance calculation failed: {str(e)}")
            return 0.5

    async def find_code_dependencies(self, program_name: str) -> Dict[str, Any]:
        """Find code dependencies for a given program using embeddings"""
        try:
            await self._ensure_vector_initialized()
            
            # Search for chunks related to this program
            program_chunks = await self.semantic_search(
                f"program {program_name} dependencies imports calls",
                top_k=20
            )
            
            dependencies = {
                "direct_calls": [],
                "file_dependencies": [],
                "data_dependencies": [],
                "similar_programs": []
            }
            
            for chunk in program_chunks:
                metadata = chunk.get('metadata', {})
                
                # Extract dependencies from metadata
                called_paragraphs = self._safe_extract_list(metadata, 'called_paragraphs')
                dependencies["direct_calls"].extend(called_paragraphs)
                
                files = self._safe_extract_list(metadata, 'files')
                dependencies["file_dependencies"].extend(files)
                
                field_names = self._safe_extract_list(metadata, 'field_names')
                dependencies["data_dependencies"].extend(field_names)
            
            # Find similar programs
            similar_programs = await self.search_similar_components(program_name, top_k=5)
            dependencies["similar_programs"] = similar_programs.get('similar_components', [])
            
            # Remove duplicates
            for key in ["direct_calls", "file_dependencies", "data_dependencies"]:
                dependencies[key] = list(set(dependencies[key]))
            
            return {
                "status": "success",
                "program_name": program_name,
                "dependencies": dependencies,
                "total_chunks_analyzed": len(program_chunks)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Dependency analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _safe_extract_list(self, metadata: Dict, key: str) -> List[str]:
        """Safely extract a list from metadata"""
        try:
            value = metadata.get(key, [])
            if isinstance(value, str):
                value = json.loads(value)
            if isinstance(value, list):
                return [str(item) for item in value]
            return []
        except:
            return []

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
            self.logger.error(f"❌ Functionality search failed: {str(e)}")
            return []

    async def export_embeddings(self, output_path: str) -> Dict[str, Any]:
        """Export embeddings and metadata to file"""
        try:
            await self._ensure_vector_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
                    "export_timestamp": dt.now().isoformat(),
                    "coordinator_type": "api_based"
                }
            }
            
            for row in cursor.fetchall():
                program_name, chunk_id, chunk_type, content, metadata_str, embedding_str = row
                
                export_data["embeddings"].append({
                    "program_name": program_name,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "content": content,
                    "metadata": self.safe_json_loads(metadata_str),
                    "embedding": json.loads(embedding_str)
                })
            
            export_data["metadata"]["total_count"] = len(export_data["embeddings"])
            conn.close()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "status": "success",
                "output_path": output_path,
                "total_exported": export_data["metadata"]["total_count"],
                "file_size_mb": Path(output_path).stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {str(e)}")
            return {"status": "error", "error": str(e)}

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
                    if self.faiss_index and self.faiss_available:
                        embedding_vector = np.array(embedding_data["embedding"])
                        faiss_id = self.faiss_index.ntotal
                        self.faiss_index.add(embedding_vector.reshape(1, -1).astype('float32'))
                    else:
                        faiss_id = -1
                    
                    # Add to ChromaDB
                    if self.collection and self.chromadb_available:
                        chroma_metadata = {
                            "program_name": embedding_data["program_name"],
                            "chunk_id": embedding_data["chunk_id"],
                            "chunk_type": embedding_data["chunk_type"],
                            "faiss_id": faiss_id
                        }
                        
                        # Add other metadata safely
                        for key, value in embedding_data["metadata"].items():
                            if isinstance(value, (str, int, float, bool)):
                                chroma_metadata[key] = value
                            else:
                                chroma_metadata[key] = json.dumps(value)
                        
                        self.collection.add(
                            documents=[embedding_data["content"]],
                            metadatas=[chroma_metadata],
                            ids=[f"{embedding_data['program_name']}_{embedding_data['chunk_id']}"]
                        )
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to import embedding: {str(e)}")
                    continue
            
            # Save FAISS index
            await self._save_indices_safely()
            
            return {
                "status": "success",
                "input_path": input_path,
                "total_imported": imported_count,
                "total_in_file": len(import_data["embeddings"])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Import failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # FIXED: Add missing utility methods
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['coordinator_type'] = 'api_based'
            result['agent_type'] = 'vector_index'
            result['gpu_id'] = self.gpu_id
        return result

    def cleanup(self):
        """Clean up resources"""
        try:
            self.logger.info("🧹 Cleaning up VectorIndexAgent resources...")
            
            # Clear models from memory
            if hasattr(self, 'embedding_model') and self.embedding_model:
                del self.embedding_model
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
            
            # Close ChromaDB client
            if hasattr(self, 'chroma_client') and self.chroma_client:
                try:
                    # ChromaDB clients don't need explicit closing
                    self.chroma_client = None
                except:
                    pass
            
            self.logger.info("✅ VectorIndexAgent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

# FIXED: Ensure proper module exports
__all__ = ['VectorIndexAgent', 'CodeChunk', 'LocalCodeBERTEmbeddingFunction']