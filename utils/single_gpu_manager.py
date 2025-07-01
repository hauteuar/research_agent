# single_gpu_manager.py
"""
Single GPU Manager - Locks onto the best available GPU and uses it exclusively
Much simpler and more reliable than multi-GPU coordination
"""

import torch
import subprocess
import time
import logging
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime as dt

logger = logging.getLogger(__name__)
@dataclass
class DualGPUConfig:
    """Configuration for dual GPU usage"""
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems
    min_memory_gb: float = 8.0  # Minimum memory required PER GPU
    max_memory_usage_percent: float = 85.0  # Don't use more than 85% of GPU memory PER GPU
    force_gpu_ids: Optional[List[int]] = None  # Force specific GPUs [1, 2] (replaces force_gpu_id)
    retry_on_oom: bool = True  # Retry with smaller models on OOM
    cleanup_on_exit: bool = True  # Clean up GPUs on shutdown
    
    # NEW dual GPU specific settings:
    load_balancing_strategy: str = "round_robin"  # "round_robin" or "memory_based"
    max_requests_per_gpu: int = 2  # Max concurrent requests per GPU
    gpu_assignment_strategy: str = "workload_based"  # How to assign agents to GPUs

class DualGPUManager:
    """Dual GPU Manager - Uses 2 GPUs for load distribution"""
    
    def __init__(self, config: Optional[DualGPUConfig] = None):
        self.config = config or DualGPUConfig()
        self.selected_gpus = []  # List of 2 GPUs
        self.gpu_engines = {}    # {gpu_id: engine}
        self.gpu_loads = {}      # {gpu_id: current_load}
        self.is_locked = False
        self.request_counter = 0
        
        self._initialize_gpus()
    
    def _initialize_gpus(self):
        """Find and lock 2 best available GPUs"""
        if self.config.force_gpu_ids:
            self.selected_gpus = self.config.force_gpu_ids[:2]
        else:
            self.selected_gpus = self._find_best_2_gpus()
        
        # Initialize load tracking
        for gpu_id in self.selected_gpus:
            self.gpu_loads[gpu_id] = 0
        
        self._lock_gpus()
    
    def _find_best_2_gpus(self) -> List[int]:
        """Find 2 best GPUs with most memory"""
        gpu_candidates = []
        gpu_count = torch.cuda.device_count()
        start_gpu = 1 if (gpu_count > 1 and self.config.exclude_gpu_0) else 0
        
        for gpu_id in range(start_gpu, gpu_count):
            gpu_info = self._get_gpu_info(gpu_id)
            if gpu_info and gpu_info['free_gb'] >= self.config.min_memory_gb:
                gpu_candidates.append((gpu_id, gpu_info['free_gb']))
        
        # Sort by available memory and take top 2
        gpu_candidates.sort(key=lambda x: x[1], reverse=True)
        return [gpu_id for gpu_id, _ in gpu_candidates[:2]]
    
    def get_next_available_gpu(self) -> int:
        """Get GPU with lowest current load (round-robin with load balancing)"""
        if not self.selected_gpus:
            raise RuntimeError("No GPUs available")
        
        # Simple round-robin for now
        gpu_id = self.selected_gpus[self.request_counter % len(self.selected_gpus)]
        self.request_counter += 1
        return gpu_id
    
    def get_llm_engine(self, gpu_id: int):
        """Get or create LLM engine for specific GPU"""
        if gpu_id not in self.gpu_engines:
            self.gpu_engines[gpu_id] = self._create_engine_for_gpu(gpu_id)
        return self.gpu_engines[gpu_id]
    
    def _create_engine_for_gpu(self, gpu_id: int):
        """Create LLM engine for specific GPU"""
        # Set CUDA device for this GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=1,
            max_model_len=self.config.max_tokens,
            gpu_memory_utilization=0.4,  # Conservative per GPU
            device="cuda:0",  # Always 0 after setting CUDA_VISIBLE_DEVICES
            trust_remote_code=True,
            enforce_eager=True,
            max_num_seqs=2,  # 2 concurrent requests per GPU
            enable_prefix_caching=False
        )
        
        from vllm import AsyncLLMEngine
        return AsyncLLMEngine.from_engine_args(engine_args)


