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
    
    def __init__(self, config: Optional[DualGPUOpulenceConfig] = None):
        self.config = config or DualGPUOpulenceConfig()
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


# Simple coordinator modification to use single GPU
class SingleGPUCoordinator:
    """Simplified coordinator that uses single GPU manager"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Instruct-hf", 
                 gpu_config: Optional[SingleGPUConfig] = None):
        
        self.model_name = model_name
        self.gpu_manager = SingleGPUManager(gpu_config)
        self.agents = {}
        
        # Create LLM engine once
        self.llm_engine = self.gpu_manager.get_llm_engine(model_name)
        
        logger.info("✅ Single GPU Coordinator initialized")
    
    async def process_files(self, file_paths: List, file_type: str = "auto"):
        """Process files using single GPU"""
        task_id = self.gpu_manager.start_task("file_processing")
        
        try:
            results = []
            for file_path in file_paths:
                # Process each file (your existing logic)
                result = await self._process_single_file(file_path, file_type)
                results.append(result)
            
            return {
                "status": "success",
                "results": results,
                "gpu_used": self.gpu_manager.selected_gpu
            }
            
        finally:
            self.gpu_manager.finish_task(task_id)
    
    async def _process_single_file(self, file_path, file_type):
        """Process a single file using the locked GPU"""
        # Your existing file processing logic here
        # All agents will automatically use the same GPU
        return {"status": "processed", "file": str(file_path)}
    
    def get_agent(self, agent_type: str):
        """Get agent (all use the same GPU)"""
        if agent_type not in self.agents:
            # Create agent with the shared LLM engine
            self.agents[agent_type] = self._create_agent(agent_type)
        
        return self.agents[agent_type]
    
    def _create_agent(self, agent_type: str):
        """Create agent using shared LLM engine"""
        # Your existing agent creation logic
        # All agents will use self.llm_engine
        return f"Agent_{agent_type}_on_GPU_{self.gpu_manager.selected_gpu}"
    
    def cleanup(self):
        """Clean up all resources"""
        self.gpu_manager.cleanup_gpu_memory()
    
    def get_status(self):
        """Get coordinator status"""
        return {
            "coordinator": "single_gpu",
            "gpu_manager": self.gpu_manager.get_status(),
            "active_agents": len(self.agents),
            "model": self.model_name
        }


# Usage example
def create_single_gpu_coordinator(model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
                                 exclude_gpu_0: bool = True,
                                 min_memory_gb: float = 8.0) -> SingleGPUCoordinator:
    """Create a single GPU coordinator with specified requirements"""
    
    config = SingleGPUConfig(
        exclude_gpu_0=exclude_gpu_0,
        min_memory_gb=min_memory_gb,
        max_memory_usage_percent=80.0,
        cleanup_on_exit=True
    )
    
    return SingleGPUCoordinator(model_name, config)


# Test function
def test_single_gpu_manager():
    """Test the single GPU manager"""
    print("Testing Single GPU Manager...")
    
    try:
        with SingleGPUManager() as gpu_mgr:
            print(f"✅ Locked GPU: {gpu_mgr.selected_gpu}")
            
            # Test LLM engine creation
            engine = gpu_mgr.get_llm_engine("microsoft/DialoGPT-medium")
            print("✅ LLM engine created")
            
            # Test task tracking
            task1 = gpu_mgr.start_task("test_task")
            time.sleep(1)
            gpu_mgr.finish_task(task1)
            
            # Test cleanup
            gpu_mgr.cleanup_gpu_memory()
            
            # Get status
            status = gpu_mgr.get_status()
            print(f"✅ Status: {status}")
            
        print("✅ Single GPU Manager test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_single_gpu_manager()