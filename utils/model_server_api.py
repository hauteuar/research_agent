#!/usr/bin/env python3
"""
Production-ready Multi-GPU Model Server API System for Opulence
Loads LLM models on multiple GPUs and exposes them via HTTP/REST API on different ports
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import multiprocessing
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log version information for debugging
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

# Check vLLM version compatibility
try:
    import vllm
    logger.info(f"vLLM version: {vllm.__version__}")
except:
    logger.warning("Could not determine vLLM version")

# ==================== GPU Detection and Management ====================

class GPUManager:
    """Manages GPU detection and allocation with memory usage awareness"""
    
    @staticmethod
    def get_gpu_memory_usage(gpu_id: int) -> Dict[str, Any]:
        """Get detailed memory usage for a specific GPU"""
        try:
            # Store current device
            current_device = torch.cuda.current_device()
            
            # Switch to target GPU
            torch.cuda.set_device(gpu_id)
            
            properties = torch.cuda.get_device_properties(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            
            # Use reserved memory as the "used" memory since it's more accurate
            # for detecting if GPU is busy with other processes
            memory_used = memory_reserved
            memory_free = properties.total_memory - memory_used
            
            # If reserved memory is very low, check allocated memory instead
            # This handles cases where memory is allocated but not reserved by PyTorch
            if memory_reserved < memory_allocated:
                memory_used = memory_allocated
                memory_free = properties.total_memory - memory_allocated
            
            # Calculate usage percentage based on used memory
            usage_percent = (memory_used / properties.total_memory) * 100
            
            # Try to get more accurate memory info using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                nvidia_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Use NVIDIA's more accurate memory reporting
                nvidia_used = nvidia_info.used
                nvidia_free = nvidia_info.free
                nvidia_total = nvidia_info.total
                nvidia_usage_percent = (nvidia_used / nvidia_total) * 100
                
                # Use NVIDIA data if it shows higher usage (more accurate)
                if nvidia_usage_percent > usage_percent:
                    memory_used = nvidia_used
                    memory_free = nvidia_free
                    usage_percent = nvidia_usage_percent
                    properties.total_memory = nvidia_total
                    
                logger.debug(f"GPU {gpu_id} NVIDIA-ML: {nvidia_usage_percent:.1f}% used ({nvidia_used/1024**3:.1f}GB/{nvidia_total/1024**3:.1f}GB)")
                
            except ImportError:
                logger.debug(f"pynvml not available, using PyTorch memory stats only")
            except Exception as e:
                logger.debug(f"Failed to get NVIDIA-ML stats for GPU {gpu_id}: {str(e)}")
            
            # Restore original device
            torch.cuda.set_device(current_device)
            
            # Handle multiprocessor_count attribute safely
            multiprocessor_count = getattr(properties, 'multiprocessor_count', 'unknown')
            
            # More conservative availability check
            is_heavily_used = usage_percent > 15  # Lowered threshold for better detection
            
            return {
                "gpu_id": gpu_id,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "total_memory_gb": properties.total_memory / (1024**3),
                "memory_allocated": memory_allocated,
                "memory_allocated_gb": memory_allocated / (1024**3),
                "memory_reserved": memory_reserved,
                "memory_reserved_gb": memory_reserved / (1024**3),
                "memory_used": memory_used,
                "memory_used_gb": memory_used / (1024**3),
                "memory_free": memory_free,
                "memory_free_gb": memory_free / (1024**3),
                "usage_percent": usage_percent,
                "is_heavily_used": is_heavily_used,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multiprocessor_count": multiprocessor_count
            }
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} memory usage: {str(e)}")
            return {}
    
    @staticmethod
    def get_available_gpus(memory_threshold_percent: float = 15.0, 
                          min_free_memory_gb: float = 6.0) -> List[Dict[str, Any]]:
        """Get list of available GPUs with memory usage details
        
        Args:
            memory_threshold_percent: Consider GPU unavailable if usage > this percent (lowered to 15%)
            min_free_memory_gb: Minimum free memory required in GB (increased to 6GB)
        """
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return []
        
        available_gpus = []
        gpu_count = torch.cuda.device_count()
        
        logger.info(f"Scanning {gpu_count} GPUs for availability...")
        logger.info(f"Criteria: Usage < {memory_threshold_percent}%, Free > {min_free_memory_gb}GB")
        
        for gpu_id in range(gpu_count):
            try:
                # Get memory usage info
                gpu_info = GPUManager.get_gpu_memory_usage(gpu_id)
                
                if not gpu_info:
                    continue
                
                # Check if GPU is accessible
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(gpu_id)
                
                # Try to allocate a small tensor to test accessibility
                test_tensor = torch.cuda.FloatTensor([1.0])
                del test_tensor
                torch.cuda.empty_cache()
                
                # Restore original device
                torch.cuda.set_device(current_device)
                
                # Evaluate GPU availability based on memory usage
                is_available = (
                    gpu_info["usage_percent"] <= memory_threshold_percent and
                    gpu_info["memory_free_gb"] >= min_free_memory_gb
                )
                
                gpu_info["is_available"] = is_available
                gpu_info["availability_reason"] = GPUManager._get_availability_reason(
                    gpu_info, memory_threshold_percent, min_free_memory_gb
                )
                
                available_gpus.append(gpu_info)
                
                # Log GPU status with more detail
                status = "✅ AVAILABLE" if is_available else "❌ BUSY/FULL"
                logger.info(
                    f"GPU {gpu_id} ({gpu_info['name']}): {status}"
                )
                logger.info(
                    f"  Usage: {gpu_info['usage_percent']:.1f}% "
                    f"({gpu_info['memory_used_gb']:.1f}GB used / {gpu_info['total_memory_gb']:.1f}GB total)"
                )
                logger.info(
                    f"  Free: {gpu_info['memory_free_gb']:.1f}GB"
                )
                if not is_available:
                    logger.info(f"  Reason: {gpu_info['availability_reason']}")
                
            except Exception as e:
                logger.warning(f"GPU {gpu_id} is not accessible: {str(e)}")
        
        return available_gpus
    
    @staticmethod
    def _get_availability_reason(gpu_info: Dict[str, Any], 
                                memory_threshold: float, 
                                min_free_memory: float) -> str:
        """Get human-readable reason for GPU availability status"""
        if gpu_info["is_available"]:
            return "Available for use"
        
        reasons = []
        if gpu_info["usage_percent"] > memory_threshold:
            reasons.append(f"High memory usage ({gpu_info['usage_percent']:.1f}% > {memory_threshold}%)")
        
        if gpu_info["memory_free_gb"] < min_free_memory:
            reasons.append(f"Insufficient free memory ({gpu_info['memory_free_gb']:.1f}GB < {min_free_memory}GB)")
        
        return "; ".join(reasons)
    
    @staticmethod
    def get_best_available_gpus(num_gpus_needed: int, 
                               memory_threshold_percent: float = 20.0,
                               min_free_memory_gb: float = 4.0) -> List[int]:
        """Get the best available GPUs sorted by availability and free memory
        
        Args:
            num_gpus_needed: Number of GPUs needed
            memory_threshold_percent: Consider GPU unavailable if usage > this percent  
            min_free_memory_gb: Minimum free memory required in GB
            
        Returns:
            List of GPU IDs sorted by preference (most available first)
        """
        all_gpus = GPUManager.get_available_gpus(memory_threshold_percent, min_free_memory_gb)
        
        if not all_gpus:
            logger.error("No GPUs detected or accessible")
            return []
        
        # Filter available GPUs
        available_gpus = [gpu for gpu in all_gpus if gpu["is_available"]]
        
        if len(available_gpus) >= num_gpus_needed:
            # Sort by free memory (descending) and usage (ascending)
            available_gpus.sort(key=lambda x: (-x["memory_free_gb"], x["usage_percent"]))
            selected_gpus = [gpu["gpu_id"] for gpu in available_gpus[:num_gpus_needed]]
            
            logger.info(f"Selected {len(selected_gpus)} best available GPUs: {selected_gpus}")
            for gpu_id in selected_gpus:
                gpu_info = next(gpu for gpu in available_gpus if gpu["gpu_id"] == gpu_id)
                logger.info(f"  GPU {gpu_id}: {gpu_info['memory_free_gb']:.1f}GB free, {gpu_info['usage_percent']:.1f}% used")
            
            return selected_gpus
        
        # If not enough "available" GPUs, fall back to least used ones
        logger.warning(f"Only {len(available_gpus)} GPUs meet criteria, but {num_gpus_needed} needed")
        logger.warning("Falling back to least used GPUs (may have memory constraints)")
        
        # Sort all GPUs by usage and free memory
        all_gpus.sort(key=lambda x: (x["usage_percent"], -x["memory_free_gb"]))
        fallback_gpus = [gpu["gpu_id"] for gpu in all_gpus[:num_gpus_needed]]
        
        logger.info(f"Fallback selection: GPUs {fallback_gpus}")
        for gpu_id in fallback_gpus:
            gpu_info = next(gpu for gpu in all_gpus if gpu["gpu_id"] == gpu_id)
            logger.warning(f"  GPU {gpu_id}: {gpu_info['memory_free_gb']:.1f}GB free, {gpu_info['usage_percent']:.1f}% used - {gpu_info['availability_reason']}")
        
        return fallback_gpus
    
    @staticmethod
    def allocate_gpus_for_servers(num_servers: int = 2, 
                                 memory_threshold_percent: float = None,
                                 min_free_memory_gb: float = None) -> List[List[int]]:
        """Allocate GPUs for multiple server instances based on memory availability
        
        Args:
            num_servers: Number of server instances to create
            memory_threshold_percent: GPU usage threshold (default from env or 15%)
            min_free_memory_gb: Minimum free memory per GPU (default from env or 6GB)
        """
        # Get thresholds from environment or use defaults
        if memory_threshold_percent is None:
            memory_threshold_percent = float(os.getenv("GPU_MEMORY_THRESHOLD_PERCENT", "15.0"))
        
        if min_free_memory_gb is None:
            min_free_memory_gb = float(os.getenv("GPU_MIN_FREE_MEMORY_GB", "6.0"))
        
        logger.info("=" * 60)
        logger.info("🔍 DETAILED GPU MEMORY ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"GPU allocation criteria: <{memory_threshold_percent}% usage, >{min_free_memory_gb}GB free")
        
        # Try to install pynvml for better GPU monitoring
        try:
            import pynvml
            logger.info("✅ Using NVIDIA-ML for accurate memory detection")
        except ImportError:
            logger.warning("⚠️  pynvml not found - install with: pip install nvidia-ml-py")
            logger.warning("   Using PyTorch memory stats (may be less accurate)")
        
        # Calculate total GPUs needed (assuming 1 GPU per server by default)
        gpus_per_server = int(os.getenv("GPUS_PER_SERVER", "1"))
        total_gpus_needed = num_servers * gpus_per_server
        
        # Get best available GPUs
        best_gpus = GPUManager.get_best_available_gpus(
            total_gpus_needed, 
            memory_threshold_percent, 
            min_free_memory_gb
        )
        
        if not best_gpus:
            logger.error("❌ No suitable GPUs found!")
            logger.error("💡 Try relaxing constraints:")
            logger.error(f"   GPU_MEMORY_THRESHOLD_PERCENT=30 (current: {memory_threshold_percent})")
            logger.error(f"   GPU_MIN_FREE_MEMORY_GB=3 (current: {min_free_memory_gb})")
            raise RuntimeError("No suitable GPUs found for server deployment")
        
        # Allocate GPUs to servers
        gpu_allocations = []
        
        if len(best_gpus) >= total_gpus_needed:
            # We have enough GPUs - distribute evenly
            for i in range(num_servers):
                start_idx = i * gpus_per_server
                end_idx = start_idx + gpus_per_server
                server_gpus = best_gpus[start_idx:end_idx]
                gpu_allocations.append(server_gpus)
        else:
            # Not enough GPUs - distribute what we have
            logger.warning(f"Only {len(best_gpus)} GPUs available for {num_servers} servers")
            
            if len(best_gpus) >= num_servers:
                # At least one GPU per server
                for i in range(num_servers):
                    gpu_allocations.append([best_gpus[i % len(best_gpus)]])
            else:
                # Fewer GPUs than servers - some servers will share
                for i in range(num_servers):
                    gpu_allocations.append([best_gpus[i % len(best_gpus)]])
        
        # Log allocation results
        logger.info("=" * 60)
        logger.info("📋 FINAL GPU ALLOCATION")
        logger.info("=" * 60)
        total_memory_needed = 0
        
        for i, gpu_ids in enumerate(gpu_allocations):
            server_id = f"server_{i}"
            gpu_info_list = []
            server_memory_needed = 0
            
            for gpu_id in gpu_ids:
                gpu_info = GPUManager.get_gpu_memory_usage(gpu_id)
                gpu_info_list.append(f"GPU{gpu_id}({gpu_info.get('memory_free_gb', 0):.1f}GB free)")
                server_memory_needed += 3.0  # Estimate 3GB per model (conservative)
            
            total_memory_needed += server_memory_needed
            logger.info(f"🖥️  {server_id}: {gpu_ids}")
            logger.info(f"   Available: {', '.join(gpu_info_list)}")
            logger.info(f"   Estimated need: {server_memory_needed:.1f}GB")
        
        logger.info(f"📊 Total estimated memory needed: {total_memory_needed:.1f}GB")
        logger.info("=" * 60)
        
        return gpu_allocations

# ==================== Configuration Classes ====================

@dataclass
class ModelServerConfig:
    """Configuration for the model server"""
    # Model configuration - Conservative settings for low memory
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    model_path: Optional[str] = None
    gpu_memory_utilization: float = 0.6  # Reduced from 0.9
    max_model_len: int = 2048  # Reduced from 4096
    tensor_parallel_size: int = 1
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8100
    workers: int = 1
    timeout: int = 300
    server_id: str = "server_0"
    
    # GPU configuration
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 128
    max_waiting_requests: int = 256
    
    # Performance settings
    enable_batching: bool = True
    max_batch_size: int = 32
    streaming_timeout: float = 30.0
    
    # Security and rate limiting
    max_tokens_per_request: int = 2048
    rate_limit_requests_per_minute: int = 1000
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    request_logging: bool = True
    
    @classmethod
    def from_env(cls, server_id: str = "server_0", port: int = 8100, gpu_ids: List[int] = None) -> 'ModelServerConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Server-specific settings
        config.server_id = server_id
        config.port = port
        config.gpu_ids = gpu_ids or [0]
        
        # Update tensor parallel size based on GPU count
        config.tensor_parallel_size = len(config.gpu_ids)
        
        # Model settings
        config.model_name = os.getenv("MODEL_NAME", config.model_name)
        config.model_path = os.getenv("MODEL_PATH", config.model_path)
        config.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", config.gpu_memory_utilization))
        config.max_model_len = int(os.getenv("MAX_MODEL_LEN", config.max_model_len))
        
        # Server settings
        config.host = os.getenv("HOST", config.host)
        config.workers = int(os.getenv("WORKERS", config.workers))
        config.timeout = int(os.getenv("TIMEOUT", config.timeout))
        
        # Performance settings
        config.enable_batching = os.getenv("ENABLE_BATCHING", "true").lower() == "true"
        config.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", config.max_batch_size))
        config.streaming_timeout = float(os.getenv("STREAMING_TIMEOUT", config.streaming_timeout))
        
        # Security settings
        config.max_tokens_per_request = int(os.getenv("MAX_TOKENS_PER_REQUEST", config.max_tokens_per_request))
        config.rate_limit_requests_per_minute = int(os.getenv("RATE_LIMIT_RPM", config.rate_limit_requests_per_minute))
        
        # Monitoring settings
        config.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        config.request_logging = os.getenv("REQUEST_LOGGING", "true").lower() == "true"
        
        return config

# ==================== Request/Response Models ====================

class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming response")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v, info):
        config = getattr(cls, '_config', None)
        if config and v > config.max_tokens_per_request:
            raise ValueError(f"max_tokens cannot exceed {config.max_tokens_per_request}")
        return v

class GenerationResponse(BaseModel):
    """Response model for text generation"""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    finish_reason: str = Field(..., description="Reason for completion")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model name used")
    server_id: str = Field(..., description="Server ID that processed the request")

class StreamingChunk(BaseModel):
    """Streaming response chunk"""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text chunk")
    finish_reason: Optional[str] = Field(None, description="Reason for completion if finished")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage (final chunk only)")
    server_id: str = Field(..., description="Server ID that processed the request")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")
    server_id: str = Field(..., description="Server ID")

class StatusResponse(BaseModel):
    """Status response with detailed information"""
    status: str = Field(..., description="Server status")
    model: str = Field(..., description="Loaded model name")
    gpu_info: Dict[str, Any] = Field(..., description="GPU information")
    memory_info: Dict[str, Any] = Field(..., description="Memory usage information")
    active_requests: int = Field(..., description="Number of active requests")
    total_requests: int = Field(..., description="Total requests processed")
    uptime: float = Field(..., description="Server uptime in seconds")
    server_id: str = Field(..., description="Server ID")

class MetricsResponse(BaseModel):
    """Metrics response for monitoring"""
    requests_per_second: float = Field(..., description="Current RPS")
    average_latency: float = Field(..., description="Average response latency")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_connections: int = Field(..., description="Active connections")
    error_rate: float = Field(..., description="Error rate percentage")
    server_id: str = Field(..., description="Server ID")

# ==================== GPU Model Loader ====================

class GPUModelLoader:
    """Manages model loading and GPU resources"""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.model_loaded = False
        self.gpu_info = {}
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the model and GPU resources"""
        try:
            logger.info(f"[{self.config.server_id}] Initializing model: {self.config.model_name}")
            logger.info(f"[{self.config.server_id}] Using GPUs: {self.config.gpu_ids}")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            # Verify GPU IDs
            available_gpus = torch.cuda.device_count()
            for gpu_id in self.config.gpu_ids:
                if gpu_id >= available_gpus:
                    raise ValueError(f"GPU {gpu_id} not available. Available GPUs: {available_gpus}")
            
            # Set CUDA visible devices
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.gpu_ids))
            
            # Create engine arguments - handle compatibility issues
            engine_kwargs = {
                "model": self.config.model_path or self.config.model_name,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": 0.7,  # Reduced from 0.9 to leave room for CUDA graphs
                "max_model_len": min(self.config.max_model_len, 2048),  # Reduce context length
                "trust_remote_code": True,
            }
            
            # Add optional parameters that may not exist in all vLLM versions
            try:
                # Try to add newer parameters safely
                engine_kwargs.update({
                    "enforce_eager": True,  # Use eager execution instead of CUDA graphs to save memory
                    "max_num_seqs": 16,  # Reduce batch size to save memory
                })
            except Exception as e:
                logger.warning(f"[{self.config.server_id}] Some vLLM parameters not supported in this version: {str(e)}")
                # Fallback to older parameter names if needed
                try:
                    engine_kwargs.update({
                        "disable_cuda_graph": True,  # Legacy parameter name
                        "max_num_seqs": 16,
                    })
                except Exception as e2:
                    logger.warning(f"[{self.config.server_id}] Legacy parameters also not supported: {str(e2)}")
                    logger.info(f"[{self.config.server_id}] Continuing with basic parameters only")
            
            # Create engine arguments
            engine_args = AsyncEngineArgs(**engine_kwargs)
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Collect GPU information
            self._collect_gpu_info()
            
            self.model_loaded = True
            logger.info(f"[{self.config.server_id}] Model loaded successfully on GPUs: {self.config.gpu_ids}")
            
        except Exception as e:
            logger.error(f"[{self.config.server_id}] Failed to initialize model: {str(e)}")
            # Log more detailed error info for debugging
            logger.error(f"[{self.config.server_id}] PyTorch version: {torch.__version__}")
            logger.error(f"[{self.config.server_id}] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.error(f"[{self.config.server_id}] CUDA version: {torch.version.cuda}")
                logger.error(f"[{self.config.server_id}] GPU count: {torch.cuda.device_count()}")
            raise
    
    def _collect_gpu_info(self):
        """Collect GPU information for monitoring"""
        self.gpu_info = {}
        for gpu_id in self.config.gpu_ids:
            try:
                gpu_info = GPUManager.get_gpu_memory_usage(gpu_id)
                if gpu_info:
                    self.gpu_info[f"gpu_{gpu_id}"] = gpu_info
            except Exception as e:
                logger.warning(f"[{self.config.server_id}] Failed to collect GPU {gpu_id} info: {str(e)}")
    
    async def cleanup(self):
        """Clean up GPU resources"""
        if self.engine:
            logger.info(f"[{self.config.server_id}] Cleaning up GPU resources...")
            # vLLM doesn't have explicit cleanup, but we can clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.engine = None
            self.model_loaded = False
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        if not self.gpu_info:
            return 0.0
        
        total_utilization = 0.0
        for gpu_data in self.gpu_info.values():
            total_utilization += gpu_data.get("utilization", 0.0)
        
        return total_utilization / len(self.gpu_info) if self.gpu_info else 0.0

# ==================== Generation Handler ====================

class GenerationHandler:
    """Handles text generation requests - FIXED VERSION"""
    
    def __init__(self, model_loader: GPUModelLoader, config: ModelServerConfig):
        self.model_loader = model_loader
        self.config = config
        self.active_requests = 0
        self.total_requests = 0
        self.request_times = []
        self.error_count = 0
        
    async def generate(self, request: GenerationRequest, request_id: str) -> Union[GenerationResponse, AsyncGenerator]:
        """Generate text based on the request - FIXED VERSION"""
        if not self.model_loader.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            self.active_requests += 1
            self.total_requests += 1
            start_time = time.time()
            
            # Log the incoming request for debugging
            logger.info(f"[{self.config.server_id}] 🔥 GENERATE REQUEST RECEIVED:")
            logger.info(f"[{self.config.server_id}]   Request ID: {request_id}")
            logger.info(f"[{self.config.server_id}]   Prompt length: {len(request.prompt)}")
            logger.info(f"[{self.config.server_id}]   Max tokens: {request.max_tokens}")
            logger.info(f"[{self.config.server_id}]   Temperature: {request.temperature}")
            logger.info(f"[{self.config.server_id}]   Stream: {request.stream}")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                seed=request.seed,
            )
            
            logger.info(f"[{self.config.server_id}] 🎯 Starting generation with vLLM engine...")
            
            if request.stream:
                return self._generate_stream(request, request_id, sampling_params, start_time)
            else:
                return await self._generate_complete_fixed(request, request_id, sampling_params, start_time)
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.config.server_id}] ❌ Generation error for request {request_id}: {str(e)}")
            logger.error(f"[{self.config.server_id}] ❌ Error type: {type(e).__name__}")
            import traceback
            logger.error(f"[{self.config.server_id}] ❌ Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        finally:
            self.active_requests -= 1
    
    async def _generate_complete_fixed(self, request: GenerationRequest, request_id: str, 
                                     sampling_params: SamplingParams, start_time: float) -> GenerationResponse:
        """FIXED: Generate complete response without timeout context manager issues"""
        try:
            logger.info(f"[{self.config.server_id}] 🚀 Calling vLLM engine.generate()...")
            
            # CRITICAL FIX: Use asyncio.wait_for to handle timeouts properly
            # instead of relying on vLLM's internal timeout handling
            try:
                # Call the vLLM engine with a reasonable timeout
                results = await asyncio.wait_for(
                    self._call_vllm_generate_safely(request.prompt, sampling_params, request_id),
                    timeout=300.0  # 5 minute timeout
                )
                
                logger.info(f"[{self.config.server_id}] ✅ vLLM generation completed successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"[{self.config.server_id}] ❌ Generation timed out after 5 minutes")
                raise HTTPException(status_code=408, detail="Generation request timed out")
            
            if not results:
                logger.error(f"[{self.config.server_id}] ❌ No results returned from vLLM")
                raise RuntimeError("No output generated")
            
            # Extract generated text
            generated_text = results.get('text', '')
            finish_reason = results.get('finish_reason', 'stop')
            prompt_tokens = results.get('prompt_tokens', 0)
            completion_tokens = results.get('completion_tokens', 0)
            
            logger.info(f"[{self.config.server_id}] 📄 Generated text length: {len(generated_text)}")
            logger.info(f"[{self.config.server_id}] 📊 Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
            
            # Record timing
            end_time = time.time()
            duration = end_time - start_time
            self.request_times.append(duration)
            
            logger.info(f"[{self.config.server_id}] ⏱️  Generation completed in {duration:.2f} seconds")
            
            # Keep only last 1000 request times for metrics
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
            
            response = GenerationResponse(
                id=request_id,
                text=generated_text,
                prompt=request.prompt,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                created=int(time.time()),
                model=self.config.model_name,
                server_id=self.config.server_id
            )
            
            logger.info(f"[{self.config.server_id}] 🎉 Response prepared successfully")
            return response
            
        except Exception as e:
            logger.error(f"[{self.config.server_id}] ❌ Complete generation error: {str(e)}")
            logger.error(f"[{self.config.server_id}] ❌ Error occurred at: {time.time() - start_time:.2f}s into request")
            raise
    
    async def _call_vllm_generate_safely(self, prompt: str, sampling_params: SamplingParams, request_id: str) -> Dict[str, Any]:
        """FIXED: Safely call vLLM generate without timeout context manager conflicts"""
        try:
            logger.info(f"[{self.config.server_id}] 🔧 Calling engine.generate with prompt: '{prompt[:50]}...'")
            
            # Call vLLM generate - this is where the timeout context manager error was happening
            results_generator = self.model_loader.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            )
            
            logger.info(f"[{self.config.server_id}] 🔄 Generator created, collecting results...")
            
            # Collect all results from the async generator
            final_output = None
            async for request_output in results_generator:
                logger.debug(f"[{self.config.server_id}] 📨 Received output chunk")
                final_output = request_output
            
            if not final_output:
                logger.error(f"[{self.config.server_id}] ❌ No final output received")
                raise RuntimeError("No output generated from vLLM")
            
            # Extract information from the final output
            if not hasattr(final_output, 'outputs') or not final_output.outputs:
                logger.error(f"[{self.config.server_id}] ❌ Final output has no outputs attribute")
                raise RuntimeError("Invalid output format from vLLM")
            
            output = final_output.outputs[0]
            generated_text = output.text
            finish_reason = getattr(output, 'finish_reason', 'stop')
            
            # Calculate token usage
            prompt_tokens = len(getattr(final_output, 'prompt_token_ids', []))
            completion_tokens = len(getattr(output, 'token_ids', []))
            
            logger.info(f"[{self.config.server_id}] ✅ Successfully extracted: {len(generated_text)} chars, {completion_tokens} tokens")
            
            return {
                'text': generated_text,
                'finish_reason': finish_reason,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
            
        except Exception as e:
            logger.error(f"[{self.config.server_id}] ❌ vLLM generation failed: {type(e).__name__}: {str(e)}")
            
            # Log more details for debugging
            logger.error(f"[{self.config.server_id}] ❌ Engine loaded: {self.model_loader.model_loaded}")
            logger.error(f"[{self.config.server_id}] ❌ Engine object: {type(self.model_loader.engine)}")
            
            raise
    
    async def _generate_stream(self, request: GenerationRequest, request_id: str, 
                             sampling_params: SamplingParams, start_time: float) -> AsyncGenerator:
        """FIXED: Generate streaming response"""
        try:
            logger.info(f"[{self.config.server_id}] 🌊 Starting streaming generation...")
            
            # Generate text with streaming
            results = self.model_loader.engine.generate(
                request.prompt,
                sampling_params,
                request_id=request_id
            )
            
            previous_text = ""
            chunk_count = 0
            
            async for request_output in results:
                chunk_count += 1
                logger.debug(f"[{self.config.server_id}] 📦 Streaming chunk {chunk_count}")
                
                # Extract current generated text
                current_text = request_output.outputs[0].text
                
                # Get the new text chunk
                new_text = current_text[len(previous_text):]
                
                # Create chunk response
                chunk = StreamingChunk(
                    id=request_id,
                    text=new_text,
                    finish_reason=request_output.outputs[0].finish_reason if request_output.finished else None,
                    server_id=self.config.server_id
                )
                
                # Add usage info to final chunk
                if request_output.finished:
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = len(request_output.outputs[0].token_ids)
                    chunk.usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                    
                    # Record timing
                    end_time = time.time()
                    duration = end_time - start_time
                    self.request_times.append(duration)
                    
                    logger.info(f"[{self.config.server_id}] 🏁 Streaming completed in {duration:.2f}s with {chunk_count} chunks")
                    
                    # Keep only last 1000 request times for metrics
                    if len(self.request_times) > 1000:
                        self.request_times = self.request_times[-1000:]
                
                yield f"data: {chunk.json()}\n\n"
                previous_text = current_text
                
        except Exception as e:
            logger.error(f"[{self.config.server_id}] ❌ Streaming generation error: {str(e)}")
            error_chunk = StreamingChunk(
                id=request_id,
                text="",
                finish_reason="error",
                server_id=self.config.server_id
            )
            yield f"data: {error_chunk.json()}\n\n"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        current_time = time.time()
        
        # Calculate RPS (requests in last 60 seconds)
        recent_requests = sum(1 for t in self.request_times if current_time - t < 60)
        rps = recent_requests / 60.0
        
        # Calculate average latency
        avg_latency = sum(self.request_times) / len(self.request_times) if self.request_times else 0.0
        
        # Calculate error rate
        error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0.0
        
        return {
            "requests_per_second": rps,
            "average_latency": avg_latency,
            "error_rate": error_rate,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "error_count": self.error_count
        }

# ==================== Health Monitor ====================

class HealthMonitor:
    """Monitors system health and provides status information"""
    
    def __init__(self, model_loader: GPUModelLoader, generation_handler: GenerationHandler):
        self.model_loader = model_loader
        self.generation_handler = generation_handler
        self.start_time = time.time()
        self.version = "1.0.0"
    
    def get_health(self) -> HealthResponse:
        """Get basic health status"""
        status = "healthy" if self.model_loader.model_loaded else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            version=self.version,
            uptime=time.time() - self.start_time,
            server_id=self.model_loader.config.server_id
        )
    
    def get_status(self) -> StatusResponse:
        """Get detailed status information"""
        # Update GPU info
        self.model_loader._collect_gpu_info()
        
        # Get system memory info
        memory_info = {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
            "used": psutil.virtual_memory().used
        }
        
        status = "healthy" if self.model_loader.model_loaded else "unhealthy"
        
        return StatusResponse(
            status=status,
            model=self.model_loader.config.model_name,
            gpu_info=self.model_loader.gpu_info,
            memory_info=memory_info,
            active_requests=self.generation_handler.active_requests,
            total_requests=self.generation_handler.total_requests,
            uptime=time.time() - self.start_time,
            server_id=self.model_loader.config.server_id
        )
    
    def get_metrics(self) -> MetricsResponse:
        """Get performance metrics"""
        metrics = self.generation_handler.get_metrics()
        
        return MetricsResponse(
            requests_per_second=metrics["requests_per_second"],
            average_latency=metrics["average_latency"],
            gpu_utilization=self.model_loader.get_gpu_utilization(),
            memory_usage=psutil.virtual_memory().percent,
            active_connections=metrics["active_requests"],
            error_rate=metrics["error_rate"],
            server_id=self.model_loader.config.server_id
        )

# ==================== Main FastAPI Application ====================

class ModelServer:
    """Main model server application"""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.model_loader = GPUModelLoader(config)
        self.generation_handler = GenerationHandler(self.model_loader, config)
        self.health_monitor = HealthMonitor(self.model_loader, self.generation_handler)
        self.app = None
        
    def create_app(self) -> FastAPI:
        """Create FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info(f"[{self.config.server_id}] Starting model server...")
            await self.model_loader.initialize()
            logger.info(f"[{self.config.server_id}] Model server started successfully")
            yield
            # Shutdown
            logger.info(f"[{self.config.server_id}] Shutting down model server...")
            await self.model_loader.cleanup()
            logger.info(f"[{self.config.server_id}] Model server shut down")
        
        app = FastAPI(
            title=f"Model Server API - {self.config.server_id}",
            description="Production-ready Multi-GPU Model Server for LLM inference",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request logging middleware
        if self.config.request_logging:
            @app.middleware("http")
            async def log_requests(request: Request, call_next):
                request_id = str(uuid.uuid4())
                start_time = time.time()
                
                # Log request
                logger.info(f"[{self.config.server_id}] Request {request_id}: {request.method} {request.url}")
                
                # Process request
                response = await call_next(request)
                
                # Log response
                process_time = time.time() - start_time
                logger.info(f"[{self.config.server_id}] Request {request_id} completed in {process_time:.4f}s with status {response.status_code}")
                
                return response
        
        # Configure GenerationRequest with config
        GenerationRequest._config = self.config
        
        # API Routes
        @app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text from prompt"""
            request_id = str(uuid.uuid4())
            
            if self.config.request_logging:
                logger.info(f"[{self.config.server_id}] Generation request {request_id}: {len(request.prompt)} chars, max_tokens={request.max_tokens}")
            
            if request.stream:
                return StreamingResponse(
                    self.generation_handler.generate(request, request_id),
                    media_type="text/plain"
                )
            else:
                return await self.generation_handler.generate(request, request_id)
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint for load balancers"""
            return self.health_monitor.get_health()
        
        @app.get("/status", response_model=StatusResponse)
        async def get_status():
            """Get detailed server status"""
            return self.health_monitor.get_status()
        
        @app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get performance metrics"""
            return self.health_monitor.get_metrics()
        
        @app.post("/reload")
        async def reload_model(background_tasks: BackgroundTasks):
            """Reload the model (admin endpoint)"""
            async def reload_task():
                try:
                    logger.info(f"[{self.config.server_id}] Reloading model...")
                    await self.model_loader.cleanup()
                    await self.model_loader.initialize()
                    logger.info(f"[{self.config.server_id}] Model reloaded successfully")
                except Exception as e:
                    logger.error(f"[{self.config.server_id}] Failed to reload model: {str(e)}")
            
            background_tasks.add_task(reload_task)
            return {"message": "Model reload initiated", "server_id": self.config.server_id}
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Model Server API",
                "version": "1.0.0",
                "model": self.config.model_name,
                "server_id": self.config.server_id,
                "gpu_ids": self.config.gpu_ids,
                "port": self.config.port,
                "status": "healthy" if self.model_loader.model_loaded else "loading"
            }
        
        self.app = app
        return app

# ==================== Server Launcher ====================

class ServerLauncher:
    """Launches and manages a single model server instance"""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.server = ModelServer(config)
        
    def run(self):
        """Run the server"""
        # Set up logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create app
        app = self.server.create_app()
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            timeout_keep_alive=self.config.timeout,
            access_log=self.config.request_logging,
            log_level=self.config.log_level.lower()
        )
        
        # Run server
        server = uvicorn.Server(uvicorn_config)
        logger.info(f"[{self.config.server_id}] Starting server on {self.config.host}:{self.config.port} with GPUs {self.config.gpu_ids}")
        server.run()

def run_server_instance(server_id: str, port: int, gpu_ids: List[int]):
    """Function to run a single server instance in a separate process"""
    try:
        # Create configuration for this server instance
        config = ModelServerConfig.from_env(server_id=server_id, port=port, gpu_ids=gpu_ids)
        
        # Create and run server
        launcher = ServerLauncher(config)
        launcher.run()
        
    except KeyboardInterrupt:
        logger.info(f"[{server_id}] Server interrupted by user")
    except Exception as e:
        logger.error(f"[{server_id}] Server failed to start: {str(e)}")
        sys.exit(1)

# ==================== Multi-Server Manager ====================

class MultiServerManager:
    """Manages multiple server instances across different GPUs and ports"""
    
    def __init__(self):
        self.processes = []
        self.server_configs = []
        
    def setup_servers(self, num_servers: int = 2, base_port: int = 8100):
        """Setup configuration for multiple servers with intelligent GPU allocation"""
        
        # Get configuration parameters
        memory_threshold = float(os.getenv("GPU_MEMORY_THRESHOLD_PERCENT", "20.0"))
        min_free_memory = float(os.getenv("GPU_MIN_FREE_MEMORY_GB", "4.0"))
        
        logger.info("=== GPU Memory Analysis ===")
        
        # Get detailed GPU information first
        all_gpu_info = GPUManager.get_available_gpus(memory_threshold, min_free_memory)
        
        if not all_gpu_info:
            raise RuntimeError("No GPUs detected or accessible")
        
        # Show current GPU status
        logger.info("Current GPU Status:")
        for gpu_info in all_gpu_info:
            status_icon = "✅" if gpu_info["is_available"] else "❌"
            logger.info(f"  {status_icon} GPU {gpu_info['gpu_id']}: {gpu_info['name']}")
            logger.info(f"     Memory: {gpu_info['memory_free_gb']:.1f}GB free / {gpu_info['total_memory_gb']:.1f}GB total ({gpu_info['usage_percent']:.1f}% used)")
            logger.info(f"     Status: {gpu_info['availability_reason']}")
        
        # Get GPU allocations
        try:
            gpu_allocations = GPUManager.allocate_gpus_for_servers(
                num_servers=num_servers,
                memory_threshold_percent=memory_threshold,
                min_free_memory_gb=min_free_memory
            )
        except RuntimeError as e:
            logger.error(f"GPU allocation failed: {str(e)}")
            
            # Emergency fallback - try with relaxed constraints
            logger.warning("Attempting fallback with relaxed memory constraints...")
            try:
                gpu_allocations = GPUManager.allocate_gpus_for_servers(
                    num_servers=num_servers,
                    memory_threshold_percent=50.0,  # More relaxed
                    min_free_memory_gb=2.0         # Lower memory requirement
                )
                logger.warning("Using relaxed constraints - monitor for OOM errors!")
            except Exception as fallback_error:
                raise RuntimeError(f"GPU allocation failed even with fallback: {str(fallback_error)}")
        
        logger.info("=== Server Configuration ===")
        
        # Create server configurations
        for i, gpu_ids in enumerate(gpu_allocations):
            server_id = f"server_{i}"
            port = base_port + i
            
            # Estimate memory usage for this server
            estimated_memory_gb = len(gpu_ids) * 2.0  # Rough estimate: 2GB per GPU
            
            self.server_configs.append({
                'server_id': server_id,
                'port': port,
                'gpu_ids': gpu_ids,
                'estimated_memory_gb': estimated_memory_gb
            })
            
            # Log detailed server config
            gpu_details = []
            for gpu_id in gpu_ids:
                gpu_info = next((g for g in all_gpu_info if g["gpu_id"] == gpu_id), None)
                if gpu_info:
                    gpu_details.append(f"GPU{gpu_id}({gpu_info['memory_free_gb']:.1f}GB)")
                else:
                    gpu_details.append(f"GPU{gpu_id}")
            
            logger.info(f"✅ {server_id}: Port {port}, GPUs {gpu_ids}")
            logger.info(f"   Memory: {', '.join(gpu_details)}, Est. usage: {estimated_memory_gb:.1f}GB")
        
        # Final validation
        logger.info("=== Deployment Validation ===")
        total_estimated_memory = sum(config['estimated_memory_gb'] for config in self.server_configs)
        logger.info(f"Total estimated memory usage: {total_estimated_memory:.1f}GB")
        
        # Check for potential issues
        warnings = []
        for config in self.server_configs:
            for gpu_id in config['gpu_ids']:
                gpu_info = next((g for g in all_gpu_info if g["gpu_id"] == gpu_id), None)
                if gpu_info and gpu_info['memory_free_gb'] < config['estimated_memory_gb']:
                    warnings.append(f"⚠️  {config['server_id']} may exceed available memory on GPU {gpu_id}")
        
        if warnings:
            logger.warning("Potential memory issues detected:")
            for warning in warnings:
                logger.warning(f"  {warning}")
            logger.warning("Consider reducing model size or batch size if OOM occurs")
        else:
            logger.info("✅ All servers should have sufficient memory")
        
        logger.info("=== Ready to Deploy ===")
        
        return True
    
    def start_servers(self):
        """Start all server instances in separate processes"""
        logger.info(f"Starting {len(self.server_configs)} server instances...")
        
        for config in self.server_configs:
            # Create process for each server
            process = multiprocessing.Process(
                target=run_server_instance,
                args=(config['server_id'], config['port'], config['gpu_ids']),
                name=f"ModelServer-{config['server_id']}"
            )
            
            process.start()
            self.processes.append(process)
            logger.info(f"Started {config['server_id']} on port {config['port']} with GPUs {config['gpu_ids']} (PID: {process.pid})")
    
    def wait_for_servers(self):
        """Wait for all server processes to complete"""
        try:
            for process in self.processes:
                process.join()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down servers...")
            self.stop_servers()
    
    def stop_servers(self):
        """Stop all server processes"""
        logger.info("Stopping all server instances...")
        
        for process in self.processes:
            if process.is_alive():
                logger.info(f"Terminating process {process.name} (PID: {process.pid})")
                process.terminate()
                process.join(timeout=10)
                
                if process.is_alive():
                    logger.warning(f"Force killing process {process.name} (PID: {process.pid})")
                    process.kill()
                    process.join()
        
        logger.info("All servers stopped")
    
    def get_server_status(self):
        """Get status of all server processes"""
        status = {}
        for i, process in enumerate(self.processes):
            config = self.server_configs[i]
            status[config['server_id']] = {
                'pid': process.pid,
                'alive': process.is_alive(),
                'port': config['port'],
                'gpu_ids': config['gpu_ids']
            }
        return status

# ==================== Load Balancer (Optional) ====================

class SimpleLoadBalancer:
    """Simple round-robin load balancer for multiple servers"""
    
    def __init__(self, server_ports: List[int], host: str = "localhost"):
        self.server_ports = server_ports
        self.host = host
        self.current_server = 0
        
    def get_next_server_url(self) -> str:
        """Get the next server URL using round-robin"""
        port = self.server_ports[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.server_ports)
        return f"http://{self.host}:{port}"
    
    async def health_check_servers(self) -> Dict[int, bool]:
        """Check health of all servers"""
        import aiohttp
        
        health_status = {}
        
        async with aiohttp.ClientSession() as session:
            for port in self.server_ports:
                try:
                    async with session.get(f"http://{self.host}:{port}/health", timeout=5) as response:
                        health_status[port] = response.status == 200
                except Exception:
                    health_status[port] = False
        
        return health_status

# ==================== Main Entry Point ====================

def main():
    """Main entry point with intelligent GPU allocation and auto-detection"""
    try:
        # Get configuration
        num_servers_env = os.getenv("NUM_SERVERS", "auto")
        base_port = int(os.getenv("BASE_PORT", "8100"))
        
        # Display startup banner
        logger.info("🚀 Starting Opulence Multi-GPU Model Server")
        logger.info("=" * 50)
        
        # Auto-detect optimal number of servers if not specified
        if num_servers_env.lower() == "auto" or num_servers_env == "0":
            logger.info("🔍 Auto-detecting optimal server configuration...")
            
            # Get available GPUs with current settings
            available_gpus = GPUManager.get_best_available_gpus(
                num_gpus_needed=10,  # Get all available
                memory_threshold_percent=float(os.getenv("GPU_MEMORY_THRESHOLD_PERCENT", "20.0")),
                min_free_memory_gb=float(os.getenv("GPU_MIN_FREE_MEMORY_GB", "4.0"))
            )
            
            if len(available_gpus) == 0:
                logger.warning("🔍 No GPUs meet ideal criteria, checking with relaxed constraints...")
                available_gpus = GPUManager.get_best_available_gpus(
                    num_gpus_needed=10, 
                    memory_threshold_percent=50.0, 
                    min_free_memory_gb=2.0
                )
            
            # Determine optimal server count
            if len(available_gpus) == 0:
                logger.error("❌ No GPUs available at all")
                raise RuntimeError("No GPUs available")
            elif len(available_gpus) == 1:
                num_servers = 1
                logger.info("🎯 Auto-detected: 1 server (single GPU available)")
            elif len(available_gpus) == 2:
                num_servers = 2
                logger.info("🎯 Auto-detected: 2 servers (optimal for 2 GPUs)")
            elif len(available_gpus) == 3:
                num_servers = 2  # Use 2 servers, one gets 2 GPUs
                logger.info("🎯 Auto-detected: 2 servers (3 GPUs available - uneven split)")
            elif len(available_gpus) >= 4:
                num_servers = min(4, len(available_gpus) // 2)  # Max 4 servers, 2+ GPUs each
                logger.info(f"🎯 Auto-detected: {num_servers} servers ({len(available_gpus)} GPUs available)")
            
            logger.info(f"💡 To override auto-detection, set NUM_SERVERS={num_servers}")
            
        else:
            # Manual override
            num_servers = int(num_servers_env)
            logger.info(f"📍 Manual override: {num_servers} servers requested")
        
        if num_servers == 1:
            # Single server mode
            logger.info("📍 Single Server Mode")
            
            # Get best available GPU
            best_gpus = GPUManager.get_best_available_gpus(
                num_gpus_needed=1,
                memory_threshold_percent=float(os.getenv("GPU_MEMORY_THRESHOLD_PERCENT", "20.0")),
                min_free_memory_gb=float(os.getenv("GPU_MIN_FREE_MEMORY_GB", "4.0"))
            )
            
            if not best_gpus:
                raise RuntimeError("No suitable GPUs found")
            
            logger.info(f"🎯 Selected GPU {best_gpus[0]} for single server")
            
            config = ModelServerConfig.from_env(
                server_id="server_0", 
                port=base_port, 
                gpu_ids=best_gpus[:1]
            )
            launcher = ServerLauncher(config)
            launcher.run()
            
        else:
            # Multi-server mode
            logger.info(f"🌐 Multi-Server Mode ({num_servers} servers)")
            
            # Set multiprocessing start method
            multiprocessing.set_start_method('spawn', force=True)
            
            # Create and setup multi-server manager
            manager = MultiServerManager()
            
            try:
                success = manager.setup_servers(num_servers=num_servers, base_port=base_port)
                if not success:
                    raise RuntimeError("Failed to setup servers")
                
            except Exception as e:
                logger.error(f"❌ Server setup failed: {str(e)}")
                
                # Try emergency single server mode
                logger.warning("🔄 Attempting emergency single server fallback...")
                try:
                    best_gpus = GPUManager.get_best_available_gpus(1, 50.0, 1.0)  # Very relaxed
                    if best_gpus:
                        logger.info(f"🆘 Emergency mode: Using GPU {best_gpus[0]}")
                        config = ModelServerConfig.from_env(
                            server_id="emergency_server", 
                            port=base_port, 
                            gpu_ids=best_gpus[:1]
                        )
                        launcher = ServerLauncher(config)
                        launcher.run()
                        return
                except Exception:
                    pass
                
                raise e
            
            # Display final server configuration
            logger.info("📋 Final Server Configuration:")
            for config in manager.server_configs:
                logger.info(f"  🖥️  {config['server_id']}: Port {config['port']}, GPUs {config['gpu_ids']}")
            
            # Start all servers
            logger.info("🚀 Launching all servers...")
            manager.start_servers()
            
            # Display access URLs
            logger.info("🌐 Server Access URLs:")
            for config in manager.server_configs:
                logger.info(f"  📡 {config['server_id']}: http://localhost:{config['port']}")
            
            logger.info("✅ All servers started successfully!")
            logger.info("📝 Usage examples:")
            logger.info("  curl -X POST http://localhost:8100/generate -d '{\"prompt\":\"Hello\"}' -H 'Content-Type: application/json'")
            logger.info("Press Ctrl+C to stop all servers")
            
            # Wait for servers or handle shutdown
            try:
                manager.wait_for_servers()
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown signal received")
            finally:
                logger.info("🧹 Cleaning up...")
                manager.stop_servers()
                logger.info("✅ Shutdown complete")
        
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {str(e)}")
        logger.error("💡 Try adjusting GPU_MEMORY_THRESHOLD_PERCENT or GPU_MIN_FREE_MEMORY_GB environment variables")
        logger.error("💡 Or set NUM_SERVERS=1 to force single server mode")
        sys.exit(1)

if __name__ == "__main__":
    main()