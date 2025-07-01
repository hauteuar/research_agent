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

# Add these methods to your DualGPUManager class in utils/single_gpu_manager.py

class DualGPUManager:
    """Dual GPU Manager - Uses 2 GPUs for load distribution"""
    
    def __init__(self, config: Optional[DualGPUConfig] = None):
        self.config = config or DualGPUConfig()
        self.selected_gpus = []  # List of 2 GPUs
        self.gpu_engines = {}    # {gpu_id: engine}
        self.gpu_loads = {}      # {gpu_id: current_load}
        self.gpu_info = {}       # {gpu_id: gpu_info_dict}
        self.is_locked = False
        self.request_counter = 0
        self.active_tasks = {}   # {gpu_id: [task_list]}
        self.total_tasks_processed = {}  # {gpu_id: count}
        self.start_time = time.time()
        
        self._initialize_gpus()
    
    def _initialize_gpus(self):
        """Find and lock 2 best available GPUs"""
        if self.config.force_gpu_ids:
            self.selected_gpus = self.config.force_gpu_ids[:2]
        else:
            self.selected_gpus = self._find_best_2_gpus()
        
        # Initialize tracking for each GPU
        for gpu_id in self.selected_gpus:
            self.gpu_loads[gpu_id] = 0
            self.active_tasks[gpu_id] = []
            self.total_tasks_processed[gpu_id] = 0
            self.gpu_info[gpu_id] = self._get_gpu_info(gpu_id)
        
        self._lock_gpus()
    
    def get_gpu_status(self, gpu_id: int) -> Dict[str, Any]:
        """Get status for a specific GPU"""
        if gpu_id not in self.selected_gpus:
            return {
                "gpu_id": gpu_id,
                "error": f"GPU {gpu_id} not managed by this coordinator",
                "is_locked": False,
                "memory_usage_gb": 0,
                "active_tasks": 0,
                "total_tasks_processed": 0
            }
        
        try:
            # Get current GPU info
            current_info = self._get_gpu_info(gpu_id)
            
            if current_info:
                return {
                    "gpu_id": gpu_id,
                    "is_locked": self.is_locked,
                    "memory_usage_gb": current_info.get("used_gb", 0),
                    "memory_free_gb": current_info.get("free_gb", 0),
                    "memory_total_gb": current_info.get("total_gb", 0),
                    "memory_usage_percent": current_info.get("memory_usage_percent", 0),
                    "active_tasks": len(self.active_tasks.get(gpu_id, [])),
                    "total_tasks_processed": self.total_tasks_processed.get(gpu_id, 0),
                    "current_load": self.gpu_loads.get(gpu_id, 0),
                    "has_llm_engine": gpu_id in self.gpu_engines,
                    "temperature": current_info.get("temperature", 0),
                    "power_draw": current_info.get("power_draw", 0),
                    "utilization_percent": current_info.get("utilization_percent", 0),
                    "process_count": current_info.get("process_count", 0),
                    "last_updated": current_info.get("last_updated", "unknown")
                }
            else:
                return {
                    "gpu_id": gpu_id,
                    "error": f"Could not get info for GPU {gpu_id}",
                    "is_locked": self.is_locked,
                    "memory_usage_gb": 0,
                    "active_tasks": len(self.active_tasks.get(gpu_id, [])),
                    "total_tasks_processed": self.total_tasks_processed.get(gpu_id, 0)
                }
                
        except Exception as e:
            return {
                "gpu_id": gpu_id,
                "error": str(e),
                "is_locked": self.is_locked,
                "memory_usage_gb": 0,
                "active_tasks": 0,
                "total_tasks_processed": 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status of the dual GPU manager"""
        try:
            # Get status for all GPUs
            gpu_statuses = {}
            total_memory_used = 0
            total_active_tasks = 0
            total_tasks_completed = 0
            
            for gpu_id in self.selected_gpus:
                gpu_status = self.get_gpu_status(gpu_id)
                gpu_statuses[f"gpu_{gpu_id}"] = gpu_status
                
                total_memory_used += gpu_status.get("memory_usage_gb", 0)
                total_active_tasks += gpu_status.get("active_tasks", 0)
                total_tasks_completed += gpu_status.get("total_tasks_processed", 0)
            
            return {
                "selected_gpus": self.selected_gpus,
                "is_locked": self.is_locked,
                "gpu_count": len(self.selected_gpus),
                "total_memory_used_gb": total_memory_used,
                "total_active_tasks": total_active_tasks,
                "total_tasks_processed": total_tasks_completed,
                "uptime_seconds": time.time() - self.start_time,
                "gpu_statuses": gpu_statuses,
                "has_engines": {gpu_id: gpu_id in self.gpu_engines for gpu_id in self.selected_gpus},
                "coordinator_type": "dual_gpu"
            }
            
        except Exception as e:
            return {
                "selected_gpus": self.selected_gpus,
                "is_locked": self.is_locked,
                "error": str(e),
                "coordinator_type": "dual_gpu"
            }
    
    def _get_gpu_info(self, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed GPU information - reuse from SingleGPUManager"""
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
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get GPU {gpu_id} info: {e}")
            return None
    
    def start_task(self, task_name: str, preferred_gpu: int = None) -> str:
        """Start a task on a specific GPU or auto-assign"""
        if preferred_gpu and preferred_gpu in self.selected_gpus:
            assigned_gpu = preferred_gpu
        else:
            # Auto-assign to GPU with lowest load
            assigned_gpu = min(self.selected_gpus, key=lambda gpu: len(self.active_tasks.get(gpu, [])))
        
        task_id = f"{task_name}_{assigned_gpu}_{len(self.active_tasks.get(assigned_gpu, []))}_{int(time.time())}"
        
        task_info = {
            "task_id": task_id,
            "task_name": task_name,
            "gpu_id": assigned_gpu,
            "start_time": time.time()
        }
        
        if assigned_gpu not in self.active_tasks:
            self.active_tasks[assigned_gpu] = []
        
        self.active_tasks[assigned_gpu].append(task_info)
        
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 Started task: {task_id} on GPU {assigned_gpu}")
        return task_id
    
    def finish_task(self, task_id: str):
        """Mark task as finished"""
        for gpu_id in self.selected_gpus:
            if gpu_id in self.active_tasks:
                for i, task in enumerate(self.active_tasks[gpu_id]):
                    if task["task_id"] == task_id:
                        duration = time.time() - task["start_time"]
                        self.active_tasks[gpu_id].pop(i)
                        self.total_tasks_processed[gpu_id] += 1
                        
                        logger = logging.getLogger(__name__)
                        logger.info(f"✅ Finished task: {task_id} on GPU {gpu_id} (duration: {duration:.2f}s)")
                        return
    
    def cleanup_gpu_memory(self, gpu_id: int = None):
        """Clean up GPU memory for specific GPU or all GPUs"""
        if not self.is_locked:
            return
        
        gpus_to_clean = [gpu_id] if gpu_id and gpu_id in self.selected_gpus else self.selected_gpus
        
        logger = logging.getLogger(__name__)
        logger.info(f"🧹 Cleaning up GPU memory for GPUs {gpus_to_clean}")
        
        for gpu in gpus_to_clean:
            try:
                # Set CUDA device and clean
                original_device = os.environ.get('CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Restore environment
                if original_device:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_device
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                
                # Wait for cleanup
                time.sleep(1)
                
                # Update GPU info
                self.gpu_info[gpu] = self._get_gpu_info(gpu)
                
            except Exception as e:
                logger.warning(f"GPU {gpu} cleanup failed: {e}")
    
    def _find_best_2_gpus(self) -> List[int]:
        """Find 2 best GPUs with most memory"""
        gpu_candidates = []
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        start_gpu = 1 if (gpu_count > 1 and self.config.exclude_gpu_0) else 0
        
        for gpu_id in range(start_gpu, gpu_count):
            gpu_info = self._get_gpu_info(gpu_id)
            if gpu_info and gpu_info['free_gb'] >= self.config.min_memory_gb:
                gpu_candidates.append((gpu_id, gpu_info['free_gb']))
        
        # Sort by available memory and take top 2
        gpu_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [gpu_id for gpu_id, _ in gpu_candidates[:2]]
        
        if len(selected) < 2:
            raise RuntimeError(f"Only {len(selected)} suitable GPUs found, need 2 for dual GPU operation")
        
        return selected
    
    def _lock_gpus(self):
        """Lock the selected GPUs"""
        if len(self.selected_gpus) < 2:
            raise RuntimeError("Need at least 2 GPUs for dual GPU operation")
        
        # Store original environment
        self.original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        
        # Set environment to show both GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.selected_gpus))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        self.is_locked = True
        
        logger = logging.getLogger(__name__)
        logger.info(f"🔒 Locked GPUs {self.selected_gpus} (CUDA_VISIBLE_DEVICES={self.selected_gpus})")
    
    def release_gpus(self):
        """Release GPU locks and clean up"""
        if not self.is_locked:
            return
        
        logger = logging.getLogger(__name__)
        logger.info(f"🔓 Releasing GPUs {self.selected_gpus}...")
        
        # Clean up memory for all GPUs
        for gpu_id in self.selected_gpus:
            self.cleanup_gpu_memory(gpu_id)
        
        # Clean up engines
        self.gpu_engines.clear()
        
        # Restore original environment
        if self.original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.original_cuda_visible
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Reset state
        self.is_locked = False
        self.selected_gpus = []
        self.active_tasks.clear()
        self.total_tasks_processed.clear()
        
        logger.info("✅ Dual GPUs released successfully")