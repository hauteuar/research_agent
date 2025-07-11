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
from pydantic import BaseModel, Field, validator
from vllm import AsyncLLMEngine, SamplingParams
try:
    from vllm import EngineArgs
except ImportError:
    # Fallback for older vLLM versions
    from vllm import AsyncEngineArgs as EngineArgs
from vllm.utils import random_uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== GPU Detection and Management ====================

class GPUManager:
    """Manages GPU detection and allocation"""
    
    @staticmethod
    def get_available_gpus() -> List[int]:
        """Get list of available GPU IDs"""
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return []
        
        available_gpus = []
        gpu_count = torch.cuda.device_count()
        
        for gpu_id in range(gpu_count):
            try:
                # Check if GPU is accessible
                torch.cuda.set_device(gpu_id)
                # Try to allocate a small tensor to test availability
                test_tensor = torch.cuda.FloatTensor([1.0])
                del test_tensor
                torch.cuda.empty_cache()
                available_gpus.append(gpu_id)
                logger.info(f"GPU {gpu_id} is available: {torch.cuda.get_device_name(gpu_id)}")
            except Exception as e:
                logger.warning(f"GPU {gpu_id} is not available: {str(e)}")
        
        return available_gpus
    
    @staticmethod
    def get_gpu_memory_info(gpu_id: int) -> Dict[str, Any]:
        """Get memory information for a specific GPU"""
        try:
            properties = torch.cuda.get_device_properties(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_cached = torch.cuda.memory_reserved(gpu_id)
            
            return {
                "name": properties.name,
                "total_memory": properties.total_memory,
                "memory_allocated": memory_allocated,
                "memory_cached": memory_cached,
                "memory_free": properties.total_memory - memory_cached,
                "utilization": (memory_cached / properties.total_memory) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} memory info: {str(e)}")
            return {}
    
    @staticmethod
    def allocate_gpus_for_servers(num_servers: int = 2) -> List[List[int]]:
        """Allocate GPUs for multiple server instances"""
        available_gpus = GPUManager.get_available_gpus()
        
        if len(available_gpus) < num_servers:
            logger.warning(f"Only {len(available_gpus)} GPUs available, but {num_servers} servers requested")
            # If we have fewer GPUs than servers, some servers will share GPUs
            gpu_allocations = []
            for i in range(num_servers):
                gpu_allocations.append([available_gpus[i % len(available_gpus)]])
            return gpu_allocations
        
        # Distribute GPUs evenly across servers
        gpu_allocations = []
        gpus_per_server = len(available_gpus) // num_servers
        remaining_gpus = len(available_gpus) % num_servers
        
        start_idx = 0
        for i in range(num_servers):
            end_idx = start_idx + gpus_per_server
            if i < remaining_gpus:
                end_idx += 1
            
            gpu_allocations.append(available_gpus[start_idx:end_idx])
            start_idx = end_idx
        
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
    port: int = 8000
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
    def from_env(cls, server_id: str = "server_0", port: int = 8000, gpu_ids: List[int] = None) -> 'ModelServerConfig':
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
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v, values):
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
            
            # Create engine arguments with only valid parameters
            engine_args = EngineArgs(
                model=self.config.model_path or self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=0.7,  # Reduced from 0.9 to leave room for CUDA graphs
                max_model_len=min(self.config.max_model_len, 2048),  # Reduce context length
                trust_remote_code=True,
                # Disable CUDA graph to save memory (trades speed for memory)
                disable_cuda_graph=True,
                # Reduce batch size to save memory
                max_num_seqs=16,  # Reduce from default
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Collect GPU information
            self._collect_gpu_info()
            
            self.model_loaded = True
            logger.info(f"[{self.config.server_id}] Model loaded successfully on GPUs: {self.config.gpu_ids}")
            
        except Exception as e:
            logger.error(f"[{self.config.server_id}] Failed to initialize model: {str(e)}")
            raise
    
    def _collect_gpu_info(self):
        """Collect GPU information for monitoring"""
        self.gpu_info = {}
        for gpu_id in self.config.gpu_ids:
            try:
                gpu_info = GPUManager.get_gpu_memory_info(gpu_id)
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
    """Handles text generation requests"""
    
    def __init__(self, model_loader: GPUModelLoader, config: ModelServerConfig):
        self.model_loader = model_loader
        self.config = config
        self.active_requests = 0
        self.total_requests = 0
        self.request_times = []
        self.error_count = 0
        
    async def generate(self, request: GenerationRequest, request_id: str) -> Union[GenerationResponse, AsyncGenerator]:
        """Generate text based on the request"""
        if not self.model_loader.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            self.active_requests += 1
            self.total_requests += 1
            start_time = time.time()
            
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
            
            if request.stream:
                return self._generate_stream(request, request_id, sampling_params, start_time)
            else:
                return await self._generate_complete(request, request_id, sampling_params, start_time)
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.config.server_id}] Generation error for request {request_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        finally:
            self.active_requests -= 1
    
    async def _generate_complete(self, request: GenerationRequest, request_id: str, 
                               sampling_params: SamplingParams, start_time: float) -> GenerationResponse:
        """Generate complete response"""
        try:
            # Generate text
            results = self.model_loader.engine.generate(
                request.prompt,
                sampling_params,
                request_id=request_id
            )
            
            # Get the final result
            final_output = None
            async for request_output in results:
                final_output = request_output
            
            if not final_output:
                raise RuntimeError("No output generated")
            
            # Extract generated text
            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            
            # Calculate token usage
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)
            
            # Record timing
            end_time = time.time()
            self.request_times.append(end_time - start_time)
            
            # Keep only last 1000 request times for metrics
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
            
            return GenerationResponse(
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
            
        except Exception as e:
            logger.error(f"[{self.config.server_id}] Complete generation error: {str(e)}")
            raise
    
    async def _generate_stream(self, request: GenerationRequest, request_id: str, 
                             sampling_params: SamplingParams, start_time: float) -> AsyncGenerator:
        """Generate streaming response"""
        try:
            # Generate text with streaming
            results = self.model_loader.engine.generate(
                request.prompt,
                sampling_params,
                request_id=request_id
            )
            
            previous_text = ""
            async for request_output in results:
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
                    self.request_times.append(end_time - start_time)
                    
                    # Keep only last 1000 request times for metrics
                    if len(self.request_times) > 1000:
                        self.request_times = self.request_times[-1000:]
                
                yield f"data: {chunk.json()}\n\n"
                previous_text = current_text
                
        except Exception as e:
            logger.error(f"[{self.config.server_id}] Streaming generation error: {str(e)}")
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
        
    def setup_servers(self, num_servers: int = 2, base_port: int = 8000):
        """Setup configuration for multiple servers"""
        # Get available GPUs and allocate them
        gpu_allocations = GPUManager.allocate_gpus_for_servers(num_servers)
        
        if not gpu_allocations:
            raise RuntimeError("No GPUs available for server deployment")
        
        logger.info(f"Setting up {num_servers} servers with GPU allocations: {gpu_allocations}")
        
        # Create server configurations
        for i, gpu_ids in enumerate(gpu_allocations):
            server_id = f"server_{i}"
            port = base_port + i
            
            self.server_configs.append({
                'server_id': server_id,
                'port': port,
                'gpu_ids': gpu_ids
            })
            
            logger.info(f"Server {server_id} will use GPUs {gpu_ids} on port {port}")
    
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
    """Main entry point"""
    try:
        # Check if we should run in single server mode or multi-server mode
        num_servers = int(os.getenv("NUM_SERVERS", "2"))
        base_port = int(os.getenv("BASE_PORT", "8000"))
        
        if num_servers == 1:
            # Single server mode (backward compatibility)
            logger.info("Running in single server mode")
            available_gpus = GPUManager.get_available_gpus()
            if not available_gpus:
                raise RuntimeError("No GPUs available")
            
            config = ModelServerConfig.from_env(
                server_id="server_0", 
                port=base_port, 
                gpu_ids=[available_gpus[0]]
            )
            launcher = ServerLauncher(config)
            launcher.run()
        else:
            # Multi-server mode
            logger.info(f"Running in multi-server mode with {num_servers} servers")
            
            # Set multiprocessing start method
            multiprocessing.set_start_method('spawn', force=True)
            
            # Create and start multi-server manager
            manager = MultiServerManager()
            manager.setup_servers(num_servers=num_servers, base_port=base_port)
            
            # Display server configuration
            logger.info("Server Configuration:")
            for config in manager.server_configs:
                logger.info(f"  {config['server_id']}: Port {config['port']}, GPUs {config['gpu_ids']}")
            
            # Start all servers
            manager.start_servers()
            
            # Wait for servers or handle shutdown
            try:
                manager.wait_for_servers()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
            finally:
                manager.stop_servers()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()