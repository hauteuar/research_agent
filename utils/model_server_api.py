#!/usr/bin/env python3
"""
Production-ready Model Server API System for Opulence
Loads LLM models on GPUs and exposes them via HTTP/REST API
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
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

# ==================== Configuration Classes ====================

@dataclass
class ModelServerConfig:
    """Configuration for the model server"""
    # Model configuration
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    model_path: Optional[str] = None
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 300
    
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
    def from_env(cls) -> 'ModelServerConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Model settings
        config.model_name = os.getenv("MODEL_NAME", config.model_name)
        config.model_path = os.getenv("MODEL_PATH", config.model_path)
        config.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", config.gpu_memory_utilization))
        config.max_model_len = int(os.getenv("MAX_MODEL_LEN", config.max_model_len))
        config.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", config.tensor_parallel_size))
        
        # Server settings
        config.host = os.getenv("HOST", config.host)
        config.port = int(os.getenv("PORT", config.port))
        config.workers = int(os.getenv("WORKERS", config.workers))
        config.timeout = int(os.getenv("TIMEOUT", config.timeout))
        
        # GPU settings
        gpu_ids_str = os.getenv("GPU_IDS", "0")
        config.gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
        config.batch_size = int(os.getenv("BATCH_SIZE", config.batch_size))
        config.max_waiting_requests = int(os.getenv("MAX_WAITING_REQUESTS", config.max_waiting_requests))
        
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

class StreamingChunk(BaseModel):
    """Streaming response chunk"""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text chunk")
    finish_reason: Optional[str] = Field(None, description="Reason for completion if finished")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage (final chunk only)")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")

class StatusResponse(BaseModel):
    """Status response with detailed information"""
    status: str = Field(..., description="Server status")
    model: str = Field(..., description="Loaded model name")
    gpu_info: Dict[str, Any] = Field(..., description="GPU information")
    memory_info: Dict[str, Any] = Field(..., description="Memory usage information")
    active_requests: int = Field(..., description="Number of active requests")
    total_requests: int = Field(..., description="Total requests processed")
    uptime: float = Field(..., description="Server uptime in seconds")

class MetricsResponse(BaseModel):
    """Metrics response for monitoring"""
    requests_per_second: float = Field(..., description="Current RPS")
    average_latency: float = Field(..., description="Average response latency")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_connections: int = Field(..., description="Active connections")
    error_rate: float = Field(..., description="Error rate percentage")

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
            logger.info(f"Initializing model: {self.config.model_name}")
            
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
            
            # Create engine arguments
            engine_args = EngineArgs(
                model=self.config.model_path or self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
                # Remove invalid parameters that don't exist in EngineArgs
                # disable_log_stats=False,  # Not a valid parameter
                # disable_log_requests=True,  # Not a valid parameter
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Collect GPU information
            self._collect_gpu_info()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on GPUs: {self.config.gpu_ids}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def _collect_gpu_info(self):
        """Collect GPU information for monitoring"""
        self.gpu_info = {}
        for gpu_id in self.config.gpu_ids:
            try:
                properties = torch.cuda.get_device_properties(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id)
                memory_cached = torch.cuda.memory_reserved(gpu_id)
                
                self.gpu_info[f"gpu_{gpu_id}"] = {
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory": properties.total_memory,
                    "memory_allocated": memory_allocated,
                    "memory_cached": memory_cached,
                    "memory_free": properties.total_memory - memory_cached,
                    "utilization": (memory_cached / properties.total_memory) * 100
                }
            except Exception as e:
                logger.warning(f"Failed to collect GPU {gpu_id} info: {str(e)}")
    
    async def cleanup(self):
        """Clean up GPU resources"""
        if self.engine:
            logger.info("Cleaning up GPU resources...")
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
            logger.error(f"Generation error for request {request_id}: {str(e)}")
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
                model=self.config.model_name
            )
            
        except Exception as e:
            logger.error(f"Complete generation error: {str(e)}")
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
                    finish_reason=request_output.outputs[0].finish_reason if request_output.finished else None
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
            logger.error(f"Streaming generation error: {str(e)}")
            error_chunk = StreamingChunk(
                id=request_id,
                text="",
                finish_reason="error"
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
            uptime=time.time() - self.start_time
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
            uptime=time.time() - self.start_time
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
            error_rate=metrics["error_rate"]
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
            logger.info("Starting model server...")
            await self.model_loader.initialize()
            logger.info("Model server started successfully")
            yield
            # Shutdown
            logger.info("Shutting down model server...")
            await self.model_loader.cleanup()
            logger.info("Model server shut down")
        
        app = FastAPI(
            title="Model Server API",
            description="Production-ready Model Server for LLM inference",
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
                logger.info(f"Request {request_id}: {request.method} {request.url}")
                
                # Process request
                response = await call_next(request)
                
                # Log response
                process_time = time.time() - start_time
                logger.info(f"Request {request_id} completed in {process_time:.4f}s with status {response.status_code}")
                
                return response
        
        # Configure GenerationRequest with config
        GenerationRequest._config = self.config
        
        # API Routes
        @app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text from prompt"""
            request_id = str(uuid.uuid4())
            
            if self.config.request_logging:
                logger.info(f"Generation request {request_id}: {len(request.prompt)} chars, max_tokens={request.max_tokens}")
            
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
                    logger.info("Reloading model...")
                    await self.model_loader.cleanup()
                    await self.model_loader.initialize()
                    logger.info("Model reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload model: {str(e)}")
            
            background_tasks.add_task(reload_task)
            return {"message": "Model reload initiated"}
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Model Server API",
                "version": "1.0.0",
                "model": self.config.model_name,
                "status": "healthy" if self.model_loader.model_loaded else "loading"
            }
        
        self.app = app
        return app

# ==================== Server Launcher ====================

class ServerLauncher:
    """Launches and manages the model server"""
    
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
        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        server.run()

# ==================== Main Entry Point ====================

def main():
    """Main entry point"""
    try:
        # Load configuration
        config = ModelServerConfig.from_env()
        
        # Create and run server
        launcher = ServerLauncher(config)
        launcher.run()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()