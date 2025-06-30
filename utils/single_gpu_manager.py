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
class SingleGPUConfig:
    """Configuration for single GPU usage"""
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems
    min_memory_gb: float = 8.0  # Minimum memory required
    max_memory_usage_percent: float = 85.0  # Don't use more than 85% of GPU memory
    force_gpu_id: Optional[int] = None  # Force specific GPU (overrides selection)
    retry_on_oom: bool = True  # Retry with smaller models on OOM
    cleanup_on_exit: bool = True  # Clean up GPU on shutdown

class SingleGPUManager:
    """Single GPU Manager - Lock one GPU and use it for everything"""
    
    def __init__(self, config: Optional[SingleGPUConfig] = None):
        self.config = config or SingleGPUConfig()
        self.selected_gpu = None
        self.gpu_info = {}
        self.is_locked = False
        self.original_cuda_visible = None
        self.llm_engine = None
        self.active_tasks = []
        self.total_tasks_processed = 0
        self.start_time = time.time()
        
        # Initialize and lock GPU
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Find and lock the best available GPU"""
        logger.info("🔍 Initializing Single GPU Manager...")
        
        # Check if GPU is forced
        if self.config.force_gpu_id is not None:
            if self._check_gpu_availability(self.config.force_gpu_id):
                self.selected_gpu = self.config.force_gpu_id
                logger.info(f"🎯 Using forced GPU {self.selected_gpu}")
            else:
                raise RuntimeError(f"Forced GPU {self.config.force_gpu_id} is not available")
        else:
            # Find best GPU automatically
            self.selected_gpu = self._find_best_gpu()
            if self.selected_gpu is None:
                raise RuntimeError("No suitable GPU found for exclusive use")
        
        # Lock the GPU
        self._lock_gpu()
        logger.info(f"🔒 GPU {self.selected_gpu} locked for exclusive use")
    
    def _find_best_gpu(self) -> Optional[int]:
        """Find the GPU with the most available memory"""
        best_gpu = None
        best_memory = 0
        gpu_candidates = []
        
        try:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if gpu_count == 0:
                logger.error("No CUDA GPUs available")
                return None
            
            # Check all GPUs (excluding GPU 0 if configured)
            start_gpu = 1 if (gpu_count > 1 and self.config.exclude_gpu_0) else 0
            
            for gpu_id in range(start_gpu, gpu_count):
                gpu_info = self._get_gpu_info(gpu_id)
                if gpu_info:
                    free_gb = gpu_info['free_gb']
                    total_gb = gpu_info['total_gb']
                    util_percent = gpu_info['utilization_percent']
                    temp = gpu_info['temperature']
                    
                    logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free / {total_gb:.1f}GB total, "
                               f"{util_percent:.1f}% util, {temp:.1f}°C")
                    
                    # Check if GPU meets requirements
                    if (free_gb >= self.config.min_memory_gb and 
                        util_percent < 50 and  # Low utilization
                        temp < 85):  # Safe temperature
                        
                        gpu_candidates.append({
                            'gpu_id': gpu_id,
                            'free_gb': free_gb,
                            'total_gb': total_gb,
                            'score': free_gb * 10 + (100 - util_percent) * 0.1 + (100 - temp) * 0.01
                        })
            
            if not gpu_candidates:
                logger.error("No GPUs meet the minimum requirements")
                return None
            
            # Sort by score (higher is better)
            gpu_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = gpu_candidates[0]
            best_gpu = best_candidate['gpu_id']
            
            logger.info(f"✅ Selected GPU {best_gpu} with {best_candidate['free_gb']:.1f}GB free memory")
            self.gpu_info = self._get_gpu_info(best_gpu)
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"GPU selection failed: {e}")
            return None
    
    def _check_gpu_availability(self, gpu_id: int) -> bool:
        """Check if specific GPU is available"""
        gpu_info = self._get_gpu_info(gpu_id)
        if not gpu_info:
            return False
        
        return (gpu_info['free_gb'] >= self.config.min_memory_gb and
                gpu_info['utilization_percent'] < 50 and
                gpu_info['temperature'] < 85)
    
    def _get_gpu_info(self, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed GPU information"""
        try:
            # Get basic GPU info
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
            
            values = result.stdout.strip().split(',')
            if len(values) < 6:
                return None
            
            total, used, free, util, temp, power = map(float, values)
            
            # Get process count
            proc_result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            process_count = 0
            if proc_result.returncode == 0 and proc_result.stdout.strip():
                process_lines = [line.strip() for line in proc_result.stdout.strip().split('\n') if line.strip()]
                process_count = len(process_lines)
            
            return {
                'gpu_id': gpu_id,
                'total_gb': total / 1024,
                'used_gb': used / 1024,
                'free_gb': free / 1024,
                'utilization_percent': util,
                'temperature': temp,
                'power_draw': power,
                'process_count': process_count,
                'memory_usage_percent': (used / total) * 100,
                'last_updated': dt.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_id} info: {e}")
            return None
    
    def _lock_gpu(self):
        """Lock the selected GPU for exclusive use"""
        if self.selected_gpu is None:
            raise RuntimeError("No GPU selected to lock")
        
        # Store original environment
        self.original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        
        # Set environment to only show our selected GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.selected_gpu)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Set PyTorch to use device 0 (which now maps to our selected GPU)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        self.is_locked = True
        logger.info(f"🔒 GPU {self.selected_gpu} locked (CUDA_VISIBLE_DEVICES={self.selected_gpu})")
    
    def get_llm_engine(self, model_name: str, max_tokens: int = 2048, force_recreate: bool = False):
        """Get or create LLM engine on our locked GPU"""
        if not self.is_locked:
            raise RuntimeError("GPU not locked - call _lock_gpu() first")
        
        if self.llm_engine is not None and not force_recreate:
            logger.info("♻️ Reusing existing LLM engine")
            return self.llm_engine
        
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            
            # Use conservative settings for stability
            logger.info(f"🔧 Creating LLM engine for {model_name} on GPU {self.selected_gpu}")
            
            # Check available memory
            current_info = self._get_gpu_info(self.selected_gpu)
            if current_info:
                available_memory = current_info['free_gb']
                logger.info(f"📊 Available GPU memory: {available_memory:.1f}GB")
                
                # Adjust memory utilization based on available memory
                if available_memory < 6:
                    memory_utilization = 0.4
                elif available_memory < 12:
                    memory_utilization = 0.5
                else:
                    memory_utilization = 0.6
            else:
                memory_utilization = 0.4  # Conservative fallback
            
            # Create engine with conservative settings
            engine_args = AsyncEngineArgs(
                model=model_name,
                tensor_parallel_size=1,
                max_model_len=min(max_tokens, 3000),  # Conservative max length
                gpu_memory_utilization=memory_utilization,
                device="cuda:0",  # Maps to our selected GPU
                trust_remote_code=True,
                enforce_eager=True,
                disable_log_stats=False,
                quantization=None,
                load_format="auto",
                dtype="auto",
                seed=42,
                max_num_seqs=8,  # Small batch size
                enable_prefix_caching=False
            )
            
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Verify engine creation
            final_info = self._get_gpu_info(self.selected_gpu)
            if final_info:
                memory_used = self.gpu_info['free_gb'] - final_info['free_gb']
                logger.info(f"✅ LLM engine created, using {memory_used:.1f}GB GPU memory")
            
            return self.llm_engine
            
        except Exception as e:
            logger.error(f"❌ Failed to create LLM engine: {e}")
            self.llm_engine = None
            raise
    
    def start_task(self, task_name: str) -> str:
        """Register a new task"""
        task_id = f"{task_name}_{len(self.active_tasks)}_{int(time.time())}"
        self.active_tasks.append({
            'task_id': task_id,
            'task_name': task_name,
            'start_time': time.time()
        })
        logger.info(f"🚀 Started task: {task_id}")
        return task_id
    
    def finish_task(self, task_id: str):
        """Mark task as finished"""
        for i, task in enumerate(self.active_tasks):
            if task['task_id'] == task_id:
                duration = time.time() - task['start_time']
                self.active_tasks.pop(i)
                self.total_tasks_processed += 1
                logger.info(f"✅ Finished task: {task_id} (duration: {duration:.2f}s)")
                break
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory without releasing the lock"""
        if not self.is_locked:
            return
        
        try:
            logger.info(f"🧹 Cleaning up GPU {self.selected_gpu} memory...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            # Wait for cleanup
            time.sleep(2)
            
            # Check memory after cleanup
            current_info = self._get_gpu_info(self.selected_gpu)
            if current_info:
                logger.info(f"📊 After cleanup: {current_info['free_gb']:.1f}GB free")
            
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the GPU manager"""
        current_info = self._get_gpu_info(self.selected_gpu) if self.selected_gpu else {}
        
        return {
            'selected_gpu': self.selected_gpu,
            'is_locked': self.is_locked,
            'has_llm_engine': self.llm_engine is not None,
            'active_tasks': len(self.active_tasks),
            'total_tasks_processed': self.total_tasks_processed,
            'uptime_seconds': time.time() - self.start_time,
            'gpu_info': current_info,
            'active_task_details': self.active_tasks.copy(),
            'memory_usage_gb': (current_info.get('used_gb', 0) if current_info else 0)
        }
    
    def switch_to_gpu(self, new_gpu_id: int):
        """Switch to a different GPU (releases current lock)"""
        logger.info(f"🔄 Switching from GPU {self.selected_gpu} to GPU {new_gpu_id}")
        
        # Clean up current GPU
        self.cleanup_gpu_memory()
        
        # Release current LLM engine
        self.llm_engine = None
        
        # Check new GPU availability
        if not self._check_gpu_availability(new_gpu_id):
            raise RuntimeError(f"GPU {new_gpu_id} is not available for switching")
        
        # Update selection and re-lock
        self.selected_gpu = new_gpu_id
        self.gpu_info = self._get_gpu_info(new_gpu_id)
        self._lock_gpu()
        
        logger.info(f"✅ Switched to GPU {new_gpu_id}")
    
    def release_gpu(self):
        """Release the GPU lock and clean up"""
        if not self.is_locked:
            return
        
        logger.info(f"🔓 Releasing GPU {self.selected_gpu}...")
        
        # Clean up memory
        self.cleanup_gpu_memory()
        
        # Clean up LLM engine
        self.llm_engine = None
        
        # Restore original environment
        if self.original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.original_cuda_visible
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Reset state
        self.is_locked = False
        self.selected_gpu = None
        self.active_tasks.clear()
        
        logger.info("✅ GPU released successfully")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.config.cleanup_on_exit:
            self.release_gpu()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        if hasattr(self, 'is_locked') and self.is_locked and self.config.cleanup_on_exit:
            try:
                self.release_gpu()
            except:
                pass


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