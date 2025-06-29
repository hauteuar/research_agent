# optimized_gpu_manager.py
"""
Optimized GPU Manager safe for shared servers
Removes dangerous process killing and reduces monitoring overhead
"""

import torch
import logging
import subprocess
import time
import gc
import psutil
import os
import asyncio
from typing import Optional, Dict, List
import threading
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class GPUStatus:
    gpu_id: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int
    utilization: float
    is_available: bool
    our_processes_only: bool = True  # Only track our own processes

class SafeGPUForcer:
    """Safe GPU forcing without killing other users' processes"""
    
    _gpu_locks = {}
    _our_process_pids = set()  # Track only our PIDs
    
    @classmethod
    def init_gpu_locks(cls, gpu_count: int = 4):
        """Initialize locks for each GPU"""
        for i in range(gpu_count):
            if i not in cls._gpu_locks:
                cls._gpu_locks[i] = threading.RLock()  # Use RLock for nested calls
    
    @staticmethod
    def check_system_resources_light() -> Dict:
        """Lightweight system resource check"""
        try:
            # Quick checks only
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            
            return {
                'memory_percent': memory.percent,
                'our_process_memory_mb': process.memory_info().rss / (1024 * 1024),
                'critical_load': memory.percent > 95,  # Only critical threshold
                'available_memory_gb': memory.available / (1024**3)
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {'critical_load': True, 'error': str(e)}
    
    @staticmethod
    def check_gpu_memory_safe(gpu_id: int) -> Dict:
        """Safe GPU memory checking without process details"""
        try:
            # Basic memory info only - no process inspection
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True, timeout=5)  # Reduced timeout
            
            line = result.stdout.strip()
            if line:
                total, used, free, utilization = map(int, line.split(','))
                
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'free_gb': free / 1024,
                    'total_gb': total / 1024,
                    'utilization_percent': utilization,
                    'is_available': free > 2000 and utilization < 85,  # Conservative threshold
                }
        except subprocess.TimeoutExpired:
            logger.warning(f"nvidia-smi timeout for GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_id} memory: {e}")
        
        return {
            'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
            'free_gb': 0, 'total_gb': 0, 'utilization_percent': 100,
            'is_available': False
        }
    
    @staticmethod
    def safe_gpu_cleanup(gpu_id: int) -> bool:
        """Safe GPU cleanup - only our own PyTorch memory"""
        logger.info(f"Safe cleanup for GPU {gpu_id} (our process only)")
        
        try:
            # Only clean up OUR PyTorch allocations
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                # Save current device
                current_device = torch.cuda.current_device()
                
                try:
                    # Set to target GPU and clean only our allocations
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Reset our memory stats only
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats(gpu_id)
                    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                        torch.cuda.reset_accumulated_memory_stats(gpu_id)
                        
                finally:
                    # Restore original device
                    torch.cuda.set_device(current_device)
                
                # Our own garbage collection
                gc.collect()
                
                logger.info(f"Safe cleanup completed for GPU {gpu_id}")
                return True
                
        except Exception as e:
            logger.error(f"Safe GPU cleanup failed for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def find_optimal_gpu_safe(min_free_gb: float = 4.0, exclude_gpu_0: bool = True) -> Optional[int]:
        """Find optimal GPU safely without affecting other users"""
        
        gpu_candidates = []
        
        # Check available GPUs
        start_gpu = 1 if exclude_gpu_0 else 0
        for gpu_id in range(start_gpu, min(4, torch.cuda.device_count() if torch.cuda.is_available() else 0)):
            memory_info = SafeGPUForcer.check_gpu_memory_safe(gpu_id)
            
            # Simple scoring based on available memory
            free_gb = memory_info['free_gb']
            utilization = memory_info['utilization_percent']
            
            if memory_info['is_available'] and free_gb >= min_free_gb:
                # Simple score: more free memory = better
                score = free_gb - (utilization / 100.0)
                gpu_candidates.append((gpu_id, score, memory_info))
        
        # Sort by score
        gpu_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if gpu_candidates:
            best_gpu = gpu_candidates[0][0]
            best_info = gpu_candidates[0][2]
            logger.info(f"Selected GPU {best_gpu}: {best_info['free_gb']:.1f}GB free")
            return best_gpu
        
        # Fallback to GPU 0 with lower requirements
        if exclude_gpu_0:
            gpu_0_info = SafeGPUForcer.check_gpu_memory_safe(0)
            if gpu_0_info['free_gb'] >= min_free_gb / 2:
                logger.warning(f"Using GPU 0 as fallback: {gpu_0_info['free_gb']:.1f}GB free")
                return 0
        
        logger.error("No suitable GPU found")
        return None
    
    @staticmethod
    def force_gpu_environment_safe(gpu_id: int, cleanup_first: bool = False):
        """Safely set GPU environment"""
        SafeGPUForcer.init_gpu_locks()
        
        with SafeGPUForcer._gpu_locks[gpu_id]:
            if cleanup_first:
                SafeGPUForcer.safe_gpu_cleanup(gpu_id)
            
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Conservative memory settings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Set PyTorch device safely
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # 0 maps to our target GPU
                
            logger.info(f"GPU environment set to GPU {gpu_id}")

class OptimizedGPUManager:
    """Optimized GPU manager with minimal overhead"""
    
    def __init__(self, gpu_count: int = 4):
        self.gpu_count = min(gpu_count, torch.cuda.device_count() if torch.cuda.is_available() else 0)
        self.logger = logging.getLogger(__name__)
        self.gpu_stats = {}
        self.workload_queue = defaultdict(list)
        self.lock = threading.RLock()
        
        # Lightweight monitoring
        self._last_update = 0
        self._update_interval = 30  # Update every 30 seconds instead of 10
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize lightweight monitoring"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available")
            return
        
        self.logger.info(f"Initialized GPU manager with {self.gpu_count} GPUs")
        
        # Start lightweight monitoring thread
        self.monitoring_thread = threading.Thread(target=self._lightweight_monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _lightweight_monitor(self):
        """Lightweight GPU monitoring"""
        while True:
            try:
                current_time = time.time()
                
                # Only update if enough time has passed
                if current_time - self._last_update < self._update_interval:
                    time.sleep(5)
                    continue
                
                if torch.cuda.is_available():
                    # Quick update for each GPU
                    for gpu_id in range(self.gpu_count):
                        try:
                            memory_info = SafeGPUForcer.check_gpu_memory_safe(gpu_id)
                            
                            # Get GPU properties once (cached by PyTorch)
                            props = torch.cuda.get_device_properties(gpu_id)
                            
                            self.gpu_stats[gpu_id] = GPUStatus(
                                gpu_id=gpu_id,
                                name=props.name,
                                total_memory=memory_info['total_mb'] * 1024 * 1024,
                                used_memory=memory_info['used_mb'] * 1024 * 1024,
                                free_memory=memory_info['free_mb'] * 1024 * 1024,
                                utilization=memory_info['utilization_percent'],
                                is_available=memory_info['is_available']
                            )
                        except Exception as e:
                            self.logger.warning(f"GPU {gpu_id} monitoring error: {e}")
                
                self._last_update = current_time
                time.sleep(self._update_interval)  # Longer sleep
                
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_available_gpu(self) -> Optional[int]:
        """Get available GPU with caching"""
        with self.lock:
            # Use cached data if recent
            if time.time() - self._last_update > self._update_interval:
                # Force a quick update
                self._force_quick_update()
            
            if not self.gpu_stats:
                return SafeGPUForcer.find_optimal_gpu_safe(min_free_gb=2.0)
            
            available_gpus = [
                (gpu_id, status) for gpu_id, status in self.gpu_stats.items()
                if status.is_available
            ]
            
            if not available_gpus:
                # Return least utilized GPU
                if self.gpu_stats:
                    return min(self.gpu_stats.keys(), 
                              key=lambda x: self.gpu_stats[x].utilization)
                return None
            
            # Return GPU with most free memory
            return min(available_gpus, key=lambda x: x[1].used_memory)[0]
    
    def _force_quick_update(self):
        """Force a quick update of GPU stats"""
        try:
            for gpu_id in range(self.gpu_count):
                memory_info = SafeGPUForcer.check_gpu_memory_safe(gpu_id)
                if gpu_id in self.gpu_stats:
                    # Update existing status
                    status = self.gpu_stats[gpu_id]
                    status.used_memory = memory_info['used_mb'] * 1024 * 1024
                    status.free_memory = memory_info['free_mb'] * 1024 * 1024
                    status.utilization = memory_info['utilization_percent']
                    status.is_available = memory_info['is_available']
        except Exception as e:
            self.logger.warning(f"Quick update failed: {e}")
    
    def get_gpu_status_detailed(self) -> Dict[str, Any]:
        """Get detailed GPU status"""
        return {
            f"gpu_{gpu_id}": {
                "name": status.name,
                "utilization_percent": status.utilization,
                "memory_used_gb": status.used_memory / (1024**3),
                "memory_total_gb": status.total_memory / (1024**3),
                "memory_free_gb": status.free_memory / (1024**3),
                "is_available": status.is_available,
                "active_workloads": len(self.workload_queue.get(gpu_id, []))
            }
            for gpu_id, status in self.gpu_stats.items()
        }
    
    def reserve_gpu_for_workload(self, workload_type: str, preferred_gpu: int = None, 
                                duration_estimate: int = 300, allow_sharing: bool = True) -> bool:
        """Reserve GPU for workload"""
        with self.lock:
            if preferred_gpu is not None and preferred_gpu in self.gpu_stats:
                target_gpu = preferred_gpu
            else:
                target_gpu = self.get_available_gpu()
            
            if target_gpu is not None:
                self.workload_queue[target_gpu].append({
                    "type": workload_type,
                    "timestamp": time.time(),
                    "duration_estimate": duration_estimate
                })
                return True
            
            return False
    
    def release_gpu_workload(self, gpu_id: int, workload_type: str):
        """Release GPU workload"""
        with self.lock:
            if gpu_id in self.workload_queue:
                # Remove matching workload
                self.workload_queue[gpu_id] = [
                    w for w in self.workload_queue[gpu_id] 
                    if w.get("type") != workload_type
                ]
    
    def cleanup_completed_workloads(self):
        """Clean up old workloads"""
        current_time = time.time()
        with self.lock:
            for gpu_id in self.workload_queue:
                self.workload_queue[gpu_id] = [
                    w for w in self.workload_queue[gpu_id]
                    if current_time - w.get("timestamp", 0) < w.get("duration_estimate", 300)
                ]