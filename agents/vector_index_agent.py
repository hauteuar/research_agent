# agents/vector_index_agent.py
"""
Agent 2: Vector Index Builder
Creates and manages vector embeddings for code chunks using FAISS and ChromaDB
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
from datetime import datetime
import uuid
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import chromadb
from vllm import AsyncLLMEngine, SamplingParams

class VectorIndexAgent:
    """Agent for building and managing vector indices"""
    
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
             gpu_id: int = None, coordinator=None):
        self.llm_engine = llm_engine
        self.db_path = db_path or "opulence_data.db"  # ADD default value
        self.gpu_id = gpu_id
        self.coordinator = coordinator  # ADD this line
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        # ADD these lines for coordinator integration
        self._engine_created = False
        self._using_coordinator_llm = False
        
        # Initialize embedding model
        self.embedding_model_name = "microsoft/codebert-base"
        self.tokenizer = None
        self.embedding_model = None
        self.vector_dim = 768
        
        # FAISS index
        self.faiss_index = None
        self.faiss_index_path = "opulence_faiss.index"
        
        # ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "opulence_code_chunks"
        self.collection = None
        
        # Initialize components
        asyncio.create_task(self._initialize_components())

    async def _ensure_initialized(self):
        """Ensure components are initialized before use"""
        if not self._initialized:
            await self._initialize_components()
            self._initialized = True

    # ADD this new method
    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - handles both old and new vLLM API"""
        try:
            # Try new API first (with request_id)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
            return result.outputs[0].text.strip()
        except TypeError as e:
            if "request_id" in str(e):
                # Fallback to old API (without request_id)
                result = await self.llm_engine.generate(prompt, sampling_params)
                return result.outputs[0].text.strip()
            else:
                raise e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""

    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available - use coordinator first, fallback to own"""
        if self.llm_engine is not None:
            return  # Already have engine
        
        # Try to get from coordinator first
        if self.coordinator is not None:
            try:
                # Get available GPU from coordinator
                best_gpu = await self.coordinator.get_available_gpu_for_agent("vector_index")
                if best_gpu is not None:
                    # Get shared LLM engine from coordinator
                    engine = await self.coordinator.get_or_create_llm_engine(best_gpu)
                    self.llm_engine = engine
                    self.gpu_id = best_gpu
                    self._using_coordinator_llm = True
                    self.logger.info(f"VectorIndex using coordinator's LLM on GPU {best_gpu}")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to get LLM from coordinator: {e}")
        
        # Try to get from global coordinator
        if not self._engine_created:
            try:
                from opulence_coordinator import get_dynamic_coordinator
                global_coordinator = get_dynamic_coordinator()
                
                best_gpu = await global_coordinator.get_available_gpu_for_agent("vector_index")
                if best_gpu is not None:
                    engine = await global_coordinator.get_or_create_llm_engine(best_gpu)
                    self.llm_engine = engine
                    self.gpu_id = best_gpu
                    self._using_coordinator_llm = True
                    self.logger.info(f"VectorIndex using global coordinator's LLM on GPU {best_gpu}")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to get LLM from global coordinator: {e}")
        
        # Last resort: create own engine
        if not self._engine_created:
            await self._create_llm_engine()
    
    # ADD this new method
    async def _create_llm_engine(self):
        """Create own LLM engine as fallback (smaller memory footprint)"""
        try:
            from gpu_force_fix import GPUForcer
            
            best_gpu = GPUForcer.find_best_gpu_with_memory(1.5)  # Lower requirement
            if best_gpu is None:
                raise RuntimeError("No suitable GPU found for fallback LLM engine")
            
            self.logger.warning(f"VectorIndex creating fallback LLM on GPU {best_gpu}")
            
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                GPUForcer.force_gpu_environment(best_gpu)
                
                # Create smaller engine to avoid conflicts
                engine_args = GPUForcer.create_vllm_engine_args(
                    "codellama/CodeLlama-7b-Instruct-hf",
                    2048  # Smaller context
                )
                engine_args.gpu_memory_utilization = 0.3  # Use less memory
                
                from vllm import AsyncLLMEngine
                self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.gpu_id = best_gpu
                self._engine_created = True
                self._using_coordinator_llm = False
                
                self.logger.info(f"✅ VectorIndex fallback LLM created on GPU {best_gpu}")
                
            finally:
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    
        except Exception as e:
            self.logger.error(f"Failed to create fallback LLM engine: {str(e)}")
            raise
    async def create_embeddings_for_chunks(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Create embeddings for a list of chunks - MISSING METHOD"""
        try:
            await self._ensure_initialized()
            
            embeddings_created = 0
            
            for chunk_data in chunks:
                chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk_data
                
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    # Generate embedding
                    embedding = await self.embed_code_chunk(content, metadata)
                    
                    # Store in FAISS
                    faiss_id = self.faiss_index.ntotal
                    self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
                    
                    # Store in ChromaDB
                    self.collection.add(
                        embeddings=[embedding.tolist()],
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
                    embedding_id = f"{program_name}_{chunk_id_str}_embed"
                    await self._store_embedding_reference(
                        chunk_id, embedding_id, faiss_id, embedding.tolist()
                    )
                    
                    embeddings_created += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {chunk_id_str}: {str(e)}")
                    continue
            
            # Save FAISS index
            await self._save_faiss_index()
            
            result = {
                "status": "success",
                "embeddings_created": embeddings_created,
                "total_chunks": len(chunks)
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Chunk embedding creation failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})

    # FIX 4: Add missing search_similar_components method
    async def search_similar_components(self, component_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for components similar to the given component name - MISSING METHOD"""
        try:
            await self._ensure_initialized()
            
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

    # FIX 5: Add missing rebuild_index_from_chunks method
    async def rebuild_index_from_chunks(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Rebuild index from provided chunks - MISSING METHOD"""
        try:
            await self._ensure_initialized()
            
            # Clear existing indices
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Clear ChromaDB collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
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

    # FIX 6: Add helper method for shared elements
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

    # ADD this helper method
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'vector_index'
            result['using_coordinator_llm'] = self._using_coordinator_llm
        return result
    
    async def _initialize_components(self):
        """Initialize embedding model and vector databases"""
        try:
            # Wait a bit to allow coordinator initialization
            await asyncio.sleep(1)
            
            # Set device for embedding model
            device = f"cuda:{self.gpu_id}" if self.gpu_id is not None and torch.cuda.is_available() else "cpu"
            
            # Load embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.embedding_model.to(device)
            self.embedding_model.eval()
            
            # Initialize FAISS index
            if Path(self.faiss_index_path).exists():
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                self.logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
                self.logger.info("Created new FAISS index")
            
            # Initialize ChromaDB collection
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                self.logger.info(f"Loaded existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Opulence mainframe code chunks"}
                )
                self.logger.info("Created new ChromaDB collection")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector components: {str(e)}")
            raise
    
    async def embed_code_chunk(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for a code chunk"""
        try:
            # Prepare text for embedding
            text_to_embed = self._prepare_text_for_embedding(chunk_content, chunk_metadata)
            
            # Tokenize
            device = f"cuda:{self.gpu_id}" if self.gpu_id is not None and torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer(
                text_to_embed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
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
        
        # Add metadata as context
        if 'main_purpose' in metadata:
            text_parts.append(f"Purpose: {metadata['main_purpose']}")
        
        if 'field_names' in metadata and metadata['field_names']:
            text_parts.append(f"Fields: {', '.join(metadata['field_names'][:10])}")
        
        if 'operations' in metadata and metadata['operations']:
            text_parts.append(f"Operations: {', '.join(metadata['operations'][:5])}")
        
        if 'file_operations' in metadata and metadata['file_operations']:
            text_parts.append(f"File ops: {', '.join(metadata['file_operations'][:5])}")
        
        return " | ".join(text_parts)
    
    async def process_batch_embeddings(self, limit: int = None) -> Dict[str, Any]:
        """Process all unembedded chunks in batch"""
        try:
            await self._ensure_initialized()
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
                        
                        # Generate embedding
                        embedding = await self.embed_code_chunk(content, metadata)
                        
                        # Store in FAISS
                        faiss_id = self.faiss_index.ntotal
                        self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
                        
                        # Store in ChromaDB
                        self.collection.add(
                            embeddings=[embedding.tolist()],
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
    
    async def _store_embedding_reference(self, chunk_id: int, embedding_id: str, 
                                       faiss_id: int, embedding_vector: List[float]):
        """Store embedding reference in SQLite"""
        await self._ensure_initialized()
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
        """Perform semantic search on code chunks"""
        try:
            await self._ensure_initialized()
            # Generate query embedding
            query_embedding = await self.embed_code_chunk(query, {})
            
            # Search in FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(top_k * 2, self.faiss_index.ntotal)  # Get more results for filtering
            )
            
            # Get metadata from ChromaDB
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                try:
                    # Query ChromaDB for metadata
                    chroma_results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        where={"faiss_id": int(idx)},
                        n_results=1
                    )
                    
                    if chroma_results['documents']:
                        metadata = chroma_results['metadatas'][0][0]
                        
                        # Apply filters if specified
                        if filter_metadata:
                            skip = False
                            for key, value in filter_metadata.items():
                                if key in metadata and metadata[key] != value:
                                    skip = True
                                    break
                            if skip:
                                continue
                        
                        results.append({
                            "content": chroma_results['documents'][0][0],
                            "metadata": metadata,
                            "similarity_score": float(score),
                            "faiss_id": idx
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error retrieving result {idx}: {str(e)}")
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def find_similar_code_patterns(self, reference_chunk_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar code patterns to a reference chunk"""
        try:
            await self._ensure_initialized()
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
                
                # Get chunk data
                chroma_results = self.collection.query(
                    query_embeddings=[ref_embedding.tolist()],
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
        """Analyze what makes two code patterns similar - FIXED"""
        await self._ensure_llm_engine()
        await self._ensure_initialized()
        
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
        """Use LLM to enhance search query - FIXED"""
        await self._ensure_llm_engine()
        await self._ensure_initialized()
        
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
        """Use LLM to rank search results by relevance - FIXED"""
        await self._ensure_llm_engine()
        await self._ensure_initialized()
        
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
        """Calculate relevance score using LLM - FIXED"""
        await self._ensure_initialized()
        await self._ensure_llm_engine()
        
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
            await self._ensure_initialized()
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
            await self._ensure_initialized()
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
                "index_file_exists": Path(self.faiss_index_path).exists()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding statistics: {str(e)}")
            return self._add_processing_info({})
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the entire vector index from scratch"""
        try:
            await self._ensure_initialized()
            # Clear existing indices
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Clear ChromaDB collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
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
            
            return self._add_processing_info(final_result)
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {str(e)}")
            return self._add_processing_info({"status": "error", "error": str(e)})