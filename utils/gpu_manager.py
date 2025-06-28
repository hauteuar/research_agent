# gpu_manager.py - CONSOLIDATED GPU Management Module
"""
Enhanced GPU Management System with Dynamic Allocation and Error Recovery
Consolidates all GPU-related functionality into clear, non-overlapping classes
"""

import logging
import torch
import psutil
import subprocess
import time
import threading
import queue
import os
import gc
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from contextlib import contextmanager

# Initialize logger
logger = logging.getLogger(__name__)

class GPUStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OCCUPIED = "occupied"
    ERROR = "error"
    RECOVERING = "recovering"

@dataclass
class GPUInfo:
    gpu_id: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int
    utilization: float
    temperature: float
    status: GPUStatus
    process_count: int
    last_updated: float
    error_count: int = 0
    last_error: str = ""

# =====================================
# CORE GPU UTILITY CLASS - Hardware Interface
# =====================================

class GPUHardwareInterface:
    """Low-level GPU hardware interface - handles direct GPU communication"""
    
    @staticmethod
    def check_gpu_memory(gpu_id: int, retry_count: int = 2) -> dict:
        """Check GPU memory with retry logic and enhanced error handling"""
        for attempt in range(retry_count + 1):
            try:
                # Use nvidia-smi with timeout
                result = subprocess.run([
                    'nvidia-smi', 
                    f'--id={gpu_id}',
                    '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, check=True, timeout=10)
                
                line = result.stdout.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        total = int(parts[0])
                        used = int(parts[1])
                        free = int(parts[2])
                        gpu_util = float(parts[3]) if len(parts) > 3 else 0.0
                        temp = float(parts[4]) if len(parts) > 4 else 0.0
                        
                        # Get process information
                        processes = GPUHardwareInterface._get_gpu_processes(gpu_id)
                        
                        # Determine availability with more nuanced logic
                        is_available = (
                            free > 1024 and  # At least 1GB free
                            gpu_util < 90.0 and  # GPU utilization not maxed
                            len(processes) < 5 and  # Not too many processes
                            temp < 85.0  # Not overheating
                        )
                        
                        return {
                            'total_mb': total,
                            'used_mb': used,
                            'free_mb': free,
                            'free_gb': free / 1024,
                            'total_gb': total / 1024,
                            'utilization_percent': (used / total) * 100,
                            'gpu_utilization': gpu_util,
                            'temperature': temp,
                            'processes': processes,
                            'is_fragmented': used > 0 and free < 1000,
                            'is_available': is_available,
                            'is_healthy': temp < 80.0 and gpu_util < 95.0,
                            'can_share': free > 2048 and len(processes) < 3,
                            'last_check': time.time()
                        }
                
            except subprocess.TimeoutExpired:
                logger.warning(f"nvidia-smi timeout for GPU {gpu_id} (attempt {attempt + 1})")
                if attempt < retry_count:
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"GPU {gpu_id} check failed (attempt {attempt + 1}): {e}")
                if attempt < retry_count:
                    time.sleep(1)
                    continue
        
        # Return error state if all attempts failed
        return {
            'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
            'free_gb': 0, 'total_gb': 0, 'utilization_percent': 100,
            'gpu_utilization': 100, 'temperature': 0,
            'processes': [], 'is_fragmented': True, 'is_available': False,
            'is_healthy': False, 'can_share': False, 'last_check': time.time(),
            'error': 'Failed to get GPU information'
        }
    
    @staticmethod
    def _get_gpu_processes(gpu_id: int) -> List[Dict]:
        """Get processes running on specific GPU"""
        processes = []
        try:
            process_result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if process_result.stdout.strip():
                for proc_line in process_result.stdout.strip().split('\n'):
                    if proc_line.strip():
                        try:
                            pid, mem = proc_line.split(',')
                            processes.append({
                                'pid': int(pid.strip()),
                                'memory_mb': int(mem.strip())
                            })
                        except:
                            continue
        except subprocess.TimeoutExpired:
            logger.warning(f"Process query timeout for GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to get processes for GPU {gpu_id}: {e}")
        
        return processes
    
    @staticmethod
    def aggressive_gpu_cleanup(gpu_id: int) -> bool:
        """Perform aggressive GPU cleanup with multiple strategies"""
        logger.info(f"Starting aggressive cleanup for GPU {gpu_id}")
        
        try:
            # Strategy 1: Python garbage collection
            gc.collect()
            
            # Strategy 2: PyTorch cleanup
            if torch.cuda.is_available():
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(gpu_id)
                        torch.cuda.reset_accumulated_memory_stats(gpu_id)
                except Exception as e:
                    logger.warning(f"PyTorch cleanup failed: {e}")
            
            # Strategy 3: Force garbage collection again
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Strategy 4: Check and wait for cleanup to take effect
            initial_memory = GPUHardwareInterface.check_gpu_memory(gpu_id)
            time.sleep(2)
            final_memory = GPUHardwareInterface.check_gpu_memory(gpu_id)
            
            freed_mb = final_memory['free_mb'] - initial_memory['free_mb']
            logger.info(f"Cleanup freed {freed_mb}MB on GPU {gpu_id}")
            
            return final_memory['free_gb'] > initial_memory['free_gb']
            
        except Exception as e:
            logger.error(f"Aggressive cleanup failed for GPU {gpu_id}: {e}")
            return False

# =====================================
# GPU ENVIRONMENT MANAGER - Environment Setup
# =====================================

class GPUEnvironmentManager:
    """Manages GPU environment setup and configuration"""
    
    @staticmethod
    def safe_force_gpu_environment(gpu_id: int, cleanup_first: bool = True, 
                                  verify_success: bool = True) -> bool:
        """Safely force GPU environment with verification"""
        try:
            if cleanup_first:
                GPUHardwareInterface.aggressive_gpu_cleanup(gpu_id)
            
            # Check GPU availability before forcing
            memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
            if not memory_info['is_available']:
                logger.warning(f"GPU {gpu_id} is not available before forcing environment")
                if not cleanup_first:
                    GPUHardwareInterface.aggressive_gpu_cleanup(gpu_id)
                    memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
                    
                if not memory_info['is_available']:
                    logger.error(f"GPU {gpu_id} is still not available after cleanup")
                    return False
            
            # Store original environment
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # Set CUDA environment
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
            
            logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
            
            # Verify PyTorch can see the GPU
            if verify_success and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)  # 0 now maps to our target GPU
                    device_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    
                    # Test basic CUDA operation
                    test_tensor = torch.tensor([1.0], device='cuda:0')
                    test_result = test_tensor + 1
                    
                    logger.info(f"GPU environment verification successful: "
                               f"{device_count} device(s), current: {current_device}")
                    return True
                    
                except Exception as e:
                    logger.error(f"GPU environment verification failed: {e}")
                    # Restore original environment on failure
                    if original_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to force GPU environment for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def create_conservative_vllm_args(model_name: str, max_tokens: int = 2048):
        """Create very conservative VLLM engine args to avoid OOM"""
        try:
            from vllm import AsyncEngineArgs
        except ImportError:
            logger.error("vLLM not available")
            return None
        
        return AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_tokens,
            gpu_memory_utilization=0.40,  # Very conservative
            device="cuda:0",
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs
            disable_log_stats=True,
            quantization=None,
            load_format="auto",
            dtype="auto",
            seed=42,
            swap_space=1,  # 1GB swap
            max_num_seqs=16,  # Small batch size
            max_num_batched_tokens=2048,  # Limit batched tokens
            max_seq_len_to_capture=1024,  # Smaller capture length
        )

# =====================================
# GPU ALLOCATION STRATEGY - Smart GPU Selection
# =====================================

class GPUAllocationStrategy:
    """Handles intelligent GPU allocation decisions"""
    
    @staticmethod
    def find_best_gpu_with_memory(min_free_gb: float = 2.0, exclude_gpu_0: bool = True, 
                                 allow_sharing: bool = True, prioritize_health: bool = True) -> Optional[int]:
        """Enhanced GPU finding with health considerations and sharing support"""
        candidates = []
        gpu_info = {}
        
        # Check all GPUs
        gpu_range = range(1 if exclude_gpu_0 else 0, 4)
        
        for gpu_id in gpu_range:
            memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
            free_gb = memory_info['free_gb']
            gpu_info[gpu_id] = memory_info
            
            logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free, {memory_info['gpu_utilization']:.1f}% util, "
                       f"temp: {memory_info['temperature']:.1f}°C, processes: {len(memory_info['processes'])}, "
                       f"healthy: {memory_info['is_healthy']}, can_share: {memory_info['can_share']}")
            
            # Calculate suitability score
            score = GPUAllocationStrategy._calculate_gpu_score(
                memory_info, free_gb, min_free_gb, allow_sharing
            )
            
            if score > 0:
                candidates.append((gpu_id, score, free_gb, memory_info))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select best candidate that meets minimum requirements
        for gpu_id, score, free_gb, info in candidates:
            if free_gb >= min_free_gb:
                if prioritize_health and not info['is_healthy']:
                    logger.warning(f"GPU {gpu_id} has best score but is not healthy, checking next...")
                    continue
                    
                logger.info(f"Selected GPU {gpu_id} with score {score:.1f} and {free_gb:.1f}GB free")
                return gpu_id
        
        # If no GPU meets strict requirements, try with relaxed constraints
        if allow_sharing and min_free_gb > 1.0:
            logger.warning("No GPU meets strict requirements, trying with relaxed constraints...")
            return GPUAllocationStrategy.find_best_gpu_with_memory(
                min_free_gb=max(1.0, min_free_gb * 0.5),
                exclude_gpu_0=exclude_gpu_0,
                allow_sharing=True,
                prioritize_health=False
            )
        
        # Last resort: include GPU 0 if we were excluding it
        if exclude_gpu_0:
            logger.warning("No suitable GPU found excluding GPU 0, checking GPU 0 as last resort...")
            memory_info = GPUHardwareInterface.check_gpu_memory(0)
            gpu_info[0] = memory_info
            
            if memory_info['free_gb'] >= max(0.5, min_free_gb * 0.25):
                logger.warning(f"Using GPU 0 as last resort with {memory_info['free_gb']:.1f}GB free")
                return 0
        
        logger.error(f"No GPU found with at least {min_free_gb}GB free")
        return None
    
    @staticmethod
    def _calculate_gpu_score(memory_info: dict, free_gb: float, min_free_gb: float, allow_sharing: bool) -> float:
        """Calculate suitability score for GPU"""
        score = 0
        
        if memory_info['is_available']:
            score += 100
            
            # Memory score
            if free_gb >= min_free_gb:
                score += min(free_gb * 10, 100)
            
            # Health score
            if memory_info['is_healthy']:
                score += 50
            
            # Sharing capability
            if allow_sharing and memory_info['can_share']:
                score += 30
            
            # Penalize high utilization
            score -= memory_info['gpu_utilization']
            
            # Penalize many processes
            score -= len(memory_info['processes']) * 10
            
            # Temperature penalty
            if memory_info['temperature'] > 70:
                score -= (memory_info['temperature'] - 70) * 2
        
        return score

# =====================================
# WORKLOAD MANAGER - Workload Tracking and Management
# =====================================

class WorkloadManager:
    """Manages GPU workloads and resource allocation"""
    
    def __init__(self):
        self.gpu_workloads = defaultdict(list)
        self.workload_history = defaultdict(list)
        self.performance_metrics = defaultdict(dict)
        self.lock = threading.Lock()
    
    def reserve_gpu_for_workload(self, gpu_id: int, workload_type: str, 
                                duration_estimate: int = 3600, allow_sharing: bool = True) -> bool:
        """Reserve GPU for workload"""
        with self.lock:
            self.gpu_workloads[gpu_id].append({
                'workload_type': workload_type,
                'start_time': time.time(),
                'estimated_duration': duration_estimate,
                'allow_sharing': allow_sharing
            })
            return True
    
    def release_gpu_workload(self, gpu_id: int, workload_type: str):
        """Release GPU workload"""
        with self.lock:
            if gpu_id in self.gpu_workloads:
                original_count = len(self.gpu_workloads[gpu_id])
                self.gpu_workloads[gpu_id] = [
                    w for w in self.gpu_workloads[gpu_id] 
                    if w.get('workload_type') != workload_type
                ]
                released_count = original_count - len(self.gpu_workloads[gpu_id])
                if released_count > 0:
                    logger.info(f"Released {released_count} workload(s) of type {workload_type} from GPU {gpu_id}")
    
    def cleanup_completed_workloads(self):
        """Clean up completed workloads"""
        current_time = time.time()
        with self.lock:
            for gpu_id in self.gpu_workloads:
                original_count = len(self.gpu_workloads[gpu_id])
                self.gpu_workloads[gpu_id] = [
                    w for w in self.gpu_workloads[gpu_id]
                    if current_time - w.get('start_time', 0) < w.get('estimated_duration', 3600)
                ]
                cleaned_count = original_count - len(self.gpu_workloads[gpu_id])
                if cleaned_count > 0:
                    logger.debug(f"Cleaned up {cleaned_count} completed workloads from GPU {gpu_id}")
    
    def get_workload_distribution(self) -> Dict[str, List]:
        """Get workload distribution across GPUs"""
        with self.lock:
            distribution = {}
            for gpu_id in range(4):  # Assuming 4 GPUs
                distribution[f"gpu_{gpu_id}"] = self.gpu_workloads.get(gpu_id, [])
            return distribution
    
    def get_gpu_sharing_info(self) -> Dict[str, Any]:
        """Get GPU sharing information"""
        sharing_info = {}
        
        with self.lock:
            for gpu_id in range(4):
                workloads = self.gpu_workloads.get(gpu_id, [])
                sharing_info[f"gpu_{gpu_id}"] = {
                    "active_workloads": len(workloads),
                    "workload_types": [w.get('workload_type', 'unknown') for w in workloads],
                    "can_share": len(workloads) < 3,
                    "sharing_capacity": max(0, 3 - len(workloads))
                }
        
        return sharing_info

# =====================================
# MAIN GPU MANAGER - High-Level Coordination
# =====================================

class DynamicGPUManager:
    """Main GPU Manager class that coordinates all GPU operations"""
    
    def __init__(self, total_gpu_count: int = 4, memory_threshold: float = 0.80, 
                 utilization_threshold: float = 75.0):
        self.total_gpu_count = total_gpu_count
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        
        # Components
        self.workload_manager = WorkloadManager()
        
        # GPU state tracking
        self.gpu_info = {}
        self.lock = threading.Lock()
        self.last_refresh = 0
        self.refresh_interval = 5  # seconds
        
        # Error tracking
        self.gpu_error_counts = defaultdict(int)
        self.gpu_last_errors = {}
        self.recovery_attempts = defaultdict(int)
        
        # Request management
        self.request_semaphores = {}
        self.active_requests = defaultdict(int)
        
        # Initialize request semaphores
        for gpu_id in range(total_gpu_count):
            self.request_semaphores[gpu_id] = threading.Semaphore(2)
        
        # Initialize GPU info
        self._update_all_gpu_info()
    
    def _update_all_gpu_info(self):
        """Update GPU information for all GPUs"""
        try:
            for gpu_id in range(self.total_gpu_count):
                self.gpu_info[gpu_id] = self._get_gpu_info(gpu_id)
        except Exception as e:
            logger.error(f"Failed to update GPU info: {e}")
    
    def _get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get comprehensive GPU information"""
        try:
            info_dict = GPUHardwareInterface.check_gpu_memory(gpu_id)
            
            if 'error' in info_dict:
                # Handle error state
                self.gpu_error_counts[gpu_id] += 1
                self.gpu_last_errors[gpu_id] = info_dict.get('error', 'Unknown error')
                
                return GPUInfo(
                    gpu_id=gpu_id,
                    name="Error",
                    total_memory=0,
                    used_memory=0,
                    free_memory=0,
                    utilization=100.0,
                    temperature=0,
                    status=GPUStatus.ERROR,
                    process_count=0,
                    last_updated=time.time(),
                    error_count=self.gpu_error_counts[gpu_id],
                    last_error=self.gpu_last_errors[gpu_id]
                )
            
            # Reset error count on successful check
            if self.gpu_error_counts[gpu_id] > 0:
                logger.info(f"GPU {gpu_id} recovered after {self.gpu_error_counts[gpu_id]} errors")
                self.gpu_error_counts[gpu_id] = 0
            
            # Determine status
            if info_dict['gpu_utilization'] > 90:
                status = GPUStatus.OCCUPIED
            elif info_dict['gpu_utilization'] > 50 or not info_dict['is_available']:
                status = GPUStatus.BUSY
            elif info_dict['is_healthy']:
                status = GPUStatus.AVAILABLE
            else:
                status = GPUStatus.RECOVERING
            
            return GPUInfo(
                gpu_id=gpu_id,
                name=info_dict.get('name', f"GPU-{gpu_id}"),
                total_memory=info_dict['total_mb'] * 1024 * 1024,
                used_memory=info_dict['used_mb'] * 1024 * 1024,
                free_memory=info_dict['free_mb'] * 1024 * 1024,
                utilization=info_dict['gpu_utilization'],
                temperature=info_dict['temperature'],
                status=status,
                process_count=len(info_dict['processes']),
                last_updated=time.time(),
                error_count=self.gpu_error_counts[gpu_id],
                last_error=self.gpu_last_errors.get(gpu_id, "")
            )
            
        except Exception as e:
            self.gpu_error_counts[gpu_id] += 1
            self.gpu_last_errors[gpu_id] = str(e)
            logger.error(f"Failed to get GPU {gpu_id} info: {e}")
            
            return GPUInfo(
                gpu_id=gpu_id,
                name="Error",
                total_memory=0,
                used_memory=0,
                free_memory=0,
                utilization=100.0,
                temperature=0,
                status=GPUStatus.ERROR,
                process_count=0,
                last_updated=time.time(),
                error_count=self.gpu_error_counts[gpu_id],
                last_error=str(e)
            )
    
    def get_available_gpu_smart(self, preferred_gpu: Optional[int] = None, 
                              workload_type: str = "unknown", 
                              allow_sharing: bool = True,
                              exclude_gpu_0: bool = True) -> Optional[int]:
        """Smart GPU allocation with workload awareness"""
        
        with self.lock:
            # Force refresh
            try:
                self._update_all_gpu_info()
            except Exception as e:
                logger.error(f"Failed to update GPU info: {e}")
                return None
            
            # Determine memory requirements based on workload type
            if "llm_engine" in workload_type.lower():
                min_memory = 3.0
                allow_sharing = False
            elif "embedding" in workload_type.lower():
                min_memory = 1.5
            elif "analysis" in workload_type.lower():
                min_memory = 1.0
            else:
                min_memory = 2.0
            
            # Use allocation strategy
            best_gpu = GPUAllocationStrategy.find_best_gpu_with_memory(
                min_free_gb=min_memory,
                exclude_gpu_0=exclude_gpu_0,
                allow_sharing=allow_sharing,
                prioritize_health=True
            )
            
            if best_gpu is not None:
                logger.info(f"Smart allocation selected GPU {best_gpu} for {workload_type}")
                return best_gpu
            
            # Fallback logic
            return self._fallback_allocation(preferred_gpu, allow_sharing)
    
    def _fallback_allocation(self, preferred_gpu: Optional[int], allow_sharing: bool) -> Optional[int]:
        """Fallback allocation when smart allocation fails"""
        # Try preferred GPU if specified
        if preferred_gpu is not None and self._is_gpu_available(preferred_gpu):
            return preferred_gpu
        
        # Try any available GPU
        for gpu_id in range(self.total_gpu_count):
            if self._is_gpu_available(gpu_id):
                return gpu_id
        
        return None
    
    def _is_gpu_available(self, gpu_id: int) -> bool:
        """Check if GPU is available"""
        gpu_info = self.gpu_info.get(gpu_id)
        if not gpu_info:
            return False
        
        return (gpu_info.status == GPUStatus.AVAILABLE and 
                gpu_info.free_memory > 1024 * 1024 * 1024)  # 1GB
    
    def get_recommendation(self, workload_type: str) -> Dict[str, Any]:
        """Get GPU recommendation for specific workload type"""
        try:
            # Get current GPU status
            self._update_all_gpu_info()
            
            # Find best GPU
            best_gpu = self.get_available_gpu_smart(
                workload_type=workload_type,
                allow_sharing="llm_engine" not in workload_type.lower()
            )
            
            # Get alternatives
            alternatives = []
            for gpu_id in range(self.total_gpu_count):
                if gpu_id != best_gpu and self._is_gpu_available(gpu_id):
                    alternatives.append(gpu_id)
            
            confidence = 0.8 if best_gpu is not None else 0.1
            
            return {
                "recommended_gpu": best_gpu,
                "confidence": confidence,
                "reason": f"Best available GPU for {workload_type}",
                "alternative_gpus": alternatives[:2],  # Top 2 alternatives
                "workload_type": workload_type,
                "expected_performance": "good" if best_gpu is not None else "poor"
            }
            
        except Exception as e:
            logger.error(f"GPU recommendation failed for {workload_type}: {e}")
            return {
                "recommended_gpu": None,
                "confidence": 0.0,
                "reason": f"Recommendation failed: {str(e)}",
                "alternative_gpus": [],
                "workload_type": workload_type,
                "expected_performance": "unknown",
                "error": str(e)
            }
    
    def safe_request_gpu(self, gpu_id: int, workload_type: str, timeout: float = 30.0) -> bool:
        """Safely request GPU access with timeout and rate limiting"""
        try:
            semaphore = self.request_semaphores.get(gpu_id)
            if semaphore is None:
                logger.error(f"No semaphore found for GPU {gpu_id}")
                return False
            
            acquired = semaphore.acquire(timeout=timeout)
            if not acquired:
                logger.warning(f"Timeout waiting for GPU {gpu_id} access")
                return False
            
            try:
                self.active_requests[gpu_id] += 1
                logger.info(f"Acquired GPU {gpu_id} for {workload_type}")
                return True
            except Exception as e:
                logger.error(f"Error tracking GPU request: {e}")
                semaphore.release()
                return False
                
        except Exception as e:
            logger.error(f"Failed to request GPU {gpu_id}: {e}")
            return False
    
    def release_gpu_request(self, gpu_id: int, workload_type: str):
        """Release GPU request safely"""
        try:
            semaphore = self.request_semaphores.get(gpu_id)
            if semaphore is not None:
                semaphore.release()
                
            if gpu_id in self.active_requests:
                self.active_requests[gpu_id] = max(0, self.active_requests[gpu_id] - 1)
                
            logger.info(f"Released GPU {gpu_id} from {workload_type}")
                       
        except Exception as e:
            logger.error(f"Error releasing GPU {gpu_id}: {e}")
    
    def reserve_gpu_for_workload(self, workload_type: str, preferred_gpu: Optional[int] = None,
                                duration_estimate: int = 3600, allow_sharing: bool = True) -> Optional[int]:
        """Reserve GPU for workload"""
        gpu_id = self.get_available_gpu_smart(
            preferred_gpu, workload_type, allow_sharing
        )
        
        if gpu_id is not None:
            self.workload_manager.reserve_gpu_for_workload(
                gpu_id, workload_type, duration_estimate, allow_sharing
            )
        return gpu_id
    
    def release_gpu_workload(self, gpu_id: int, workload_type: str):
        """Release GPU workload"""
        self.workload_manager.release_gpu_workload(gpu_id, workload_type)
    
    def force_refresh(self):
        """Force refresh GPU status"""
        self._update_all_gpu_info()
        self.last_refresh = time.time()
        self.workload_manager.cleanup_completed_workloads()
    
    def get_gpu_status_detailed(self) -> Dict[str, Any]:
        """Get detailed GPU status"""
        status = {}
        for gpu_id, info in self.gpu_info.items():
            status[f"gpu_{gpu_id}"] = {
                'is_available': info.status == GPUStatus.AVAILABLE,
                'utilization_percent': info.utilization,
                'active_workloads': len(self.workload_manager.gpu_workloads.get(gpu_id, [])),
                'memory_free_gb': info.free_memory / (1024**3),
                'status': info.status.value,
                'process_count': info.process_count,
                'temperature': info.temperature,
                'error_count': info.error_count,
                'last_error': info.last_error
            }
        return status
    
    def get_workload_distribution(self) -> Dict[str, List]:
        """Get workload distribution across GPUs"""
        return self.workload_manager.get_workload_distribution()
    
    def get_gpu_sharing_info(self) -> Dict[str, Any]:
        """Get GPU sharing information"""
        return self.workload_manager.get_gpu_sharing_info()
    
    def cleanup_completed_workloads(self):
        """Clean up completed workloads"""
        self.workload_manager.cleanup_completed_workloads()
    
    def shutdown(self):
        """Shutdown GPU manager"""
        self.workload_manager.gpu_workloads.clear()
        self.gpu_info.clear()

# =====================================
# SAFE GPU CONTEXT MANAGER
# =====================================

@contextmanager
def SafeGPUContext(gpu_manager: DynamicGPUManager, workload_type: str,
                   preferred_gpu: Optional[int] = None, cleanup_on_exit: bool = True):
    """Context manager for safe GPU allocation and cleanup"""
    
    allocated_gpu = None
    request_acquired = False
    
    try:
        # Get GPU allocation
        allocated_gpu = gpu_manager.get_available_gpu_smart(
            preferred_gpu=preferred_gpu,
            workload_type=workload_type,
            allow_sharing="llm_engine" not in workload_type.lower()
        )
        
        if allocated_gpu is None:
            logger.error(f"No GPU available for {workload_type}")
            yield None
            return
        
        # Request safe access
        request_acquired = gpu_manager.safe_request_gpu(
            allocated_gpu, workload_type
        )
        
        if not request_acquired:
            logger.error(f"Failed to acquire safe access to GPU {allocated_gpu}")
            allocated_gpu = None
            yield None
            return
        
        # Force GPU environment
        success = GPUEnvironmentManager.safe_force_gpu_environment(
            allocated_gpu, 
            cleanup_first=True,
            verify_success=True
        )
        
        if not success:
            logger.error(f"Failed to force GPU environment for GPU {allocated_gpu}")
            gpu_manager.release_gpu_request(allocated_gpu, workload_type)
            allocated_gpu = None
            yield None
            return
        
        logger.info(f"Successfully allocated GPU {allocated_gpu} for {workload_type}")
        yield allocated_gpu
        
    except Exception as e:
        logger.error(f"GPU context setup failed: {e}")
        yield None
    
    finally:
        try:
            if allocated_gpu is not None:
                if cleanup_on_exit:
                    logger.info(f"Cleaning up GPU {allocated_gpu}")
                    GPUHardwareInterface.aggressive_gpu_cleanup(allocated_gpu)
                
                if request_acquired:
                    gpu_manager.release_gpu_request(allocated_gpu, workload_type)
                
                logger.info(f"Released GPU {allocated_gpu} from {workload_type}")
                
        except Exception as e:
            logger.error(f"GPU context cleanup failed: {e}")

# =====================================
# RECOVERY AND MONITORING UTILITIES
# =====================================

class GPURecoveryManager:
    """Handles GPU error recovery and system restoration"""
    
    @staticmethod
    def recover_from_gpu_errors() -> Dict[str, Any]:
        """Recover from common GPU errors"""
        logger.info("Starting GPU error recovery process...")
        recovery_results = {}
        
        try:
            # Check all GPUs
            logger.info("Step 1: Checking all GPU states")
            gpu_states = {}
            
            for gpu_id in range(4):
                try:
                    memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
                    gpu_states[gpu_id] = memory_info
                    logger.info(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free, "
                               f"healthy: {memory_info.get('is_healthy', False)}")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} check failed: {e}")
                    gpu_states[gpu_id] = {'error': str(e)}
            
            # Cleanup problematic GPUs
            logger.info("Step 2: Cleaning up problematic GPUs")
            for gpu_id, state in gpu_states.items():
                if 'error' in state or not state.get('is_healthy', False):
                    logger.info(f"Attempting recovery for GPU {gpu_id}")
                    success = GPUHardwareInterface.aggressive_gpu_cleanup(gpu_id)
                    recovery_results[gpu_id] = {
                        'attempted': True,
                        'success': success,
                        'initial_state': state
                    }
                    
                    if success:
                        logger.info(f"✅ GPU {gpu_id} recovery successful")
                    else:
                        logger.warning(f"❌ GPU {gpu_id} recovery failed")
                else:
                    recovery_results[gpu_id] = {
                        'attempted': False,
                        'reason': 'GPU appears healthy'
                    }
            
            # Verify recovery
            logger.info("Step 3: Verifying recovery")
            recovered_count = 0
            for gpu_id in range(4):
                try:
                    memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
                    if memory_info.get('is_available', False):
                        recovered_count += 1
                        logger.info(f"✅ GPU {gpu_id} is now available")
                        if gpu_id in recovery_results:
                            recovery_results[gpu_id]['final_state'] = 'available'
                    else:
                        logger.warning(f"❌ GPU {gpu_id} still not available")
                        if gpu_id in recovery_results:
                            recovery_results[gpu_id]['final_state'] = 'unavailable'
                except Exception as e:
                    logger.error(f"GPU {gpu_id} verification failed: {e}")
                    if gpu_id in recovery_results:
                        recovery_results[gpu_id]['final_state'] = 'error'
            
            logger.info(f"Recovery complete: {recovered_count}/4 GPUs available")
            
            return {
                'status': 'completed',
                'recovered_gpus': recovered_count,
                'total_gpus': 4,
                'recovery_details': recovery_results,
                'success_rate': recovered_count / 4
            }
            
        except Exception as e:
            logger.error(f"GPU recovery process failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'recovery_details': recovery_results
            }

class GPUMonitor:
    """Continuous GPU monitoring service"""
    
    def __init__(self, gpu_manager: DynamicGPUManager, interval: int = 30):
        self.gpu_manager = gpu_manager
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous GPU monitoring"""
        if self.monitoring:
            logger.warning("GPU monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started GPU monitoring with {self.interval}s interval")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped GPU monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._perform_health_check()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.interval)
    
    def _perform_health_check(self):
        """Perform health check on all GPUs"""
        try:
            logger.debug("=== GPU Health Check ===")
            
            # Force refresh GPU status
            self.gpu_manager.force_refresh()
            
            # Get detailed status
            gpu_status = self.gpu_manager.get_gpu_status_detailed()
            
            critical_issues = []
            warnings = []
            
            for gpu_name, status in gpu_status.items():
                gpu_id = int(gpu_name.split('_')[1])
                
                # Check for critical issues
                if status['error_count'] > 5:
                    critical_issues.append(f"GPU {gpu_id} has {status['error_count']} errors")
                
                if status['temperature'] > 85:
                    critical_issues.append(f"GPU {gpu_id} overheating: {status['temperature']:.1f}°C")
                
                # Check for warnings
                if status['utilization_percent'] > 95:
                    warnings.append(f"GPU {gpu_id} high utilization: {status['utilization_percent']:.1f}%")
                
                if status['memory_free_gb'] < 0.5:
                    warnings.append(f"GPU {gpu_id} low memory: {status['memory_free_gb']:.1f}GB free")
                
                logger.debug(f"GPU {gpu_id}: {status['memory_free_gb']:.1f}GB free, "
                            f"{status['utilization_percent']:.1f}% util, "
                            f"temp: {status['temperature']:.1f}°C, "
                            f"errors: {status['error_count']}")
            
            # Report issues
            if critical_issues:
                logger.error("🚨 CRITICAL GPU ISSUES:")
                for issue in critical_issues:
                    logger.error(f"  - {issue}")
            
            if warnings:
                logger.warning("⚠️ GPU WARNINGS:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
            
            if not critical_issues and not warnings:
                logger.debug("✅ All GPUs healthy")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")

# =====================================
# TESTING AND UTILITY FUNCTIONS
# =====================================

def test_gpu_manager():
    """Test the GPU manager functionality"""
    print("🚀 Testing Dynamic GPU Manager")
    print("=" * 50)
    
    # Create manager
    manager = DynamicGPUManager()
    
    # Test GPU status
    print("\n📊 GPU Status:")
    status = manager.get_gpu_status_detailed()
    for gpu_name, gpu_info in status.items():
        print(f"{gpu_name}: {gpu_info['status']}, "
              f"{gpu_info['memory_free_gb']:.1f}GB free, "
              f"{gpu_info['utilization_percent']:.1f}% util")
    
    # Test GPU allocation
    print("\n🔧 Testing GPU Allocation:")
    with SafeGPUContext(manager, "test_workload") as gpu_id:
        if gpu_id is not None:
            print(f"✅ Successfully allocated GPU {gpu_id}")
            
            # Test memory check
            memory_info = GPUHardwareInterface.check_gpu_memory(gpu_id)
            print(f"GPU {gpu_id} memory: {memory_info['free_gb']:.1f}GB free")
        else:
            print("❌ Failed to allocate GPU")
    
    # Test recommendations
    print("\n💡 Testing Recommendations:")
    for workload in ["llm_engine", "embedding", "analysis"]:
        rec = manager.get_recommendation(workload)
        print(f"{workload}: GPU {rec['recommended_gpu']} (confidence: {rec['confidence']:.1f})")
    
    # Cleanup
    manager.shutdown()
    print("\n✅ GPU manager test completed")

def start_gpu_monitoring(gpu_manager: DynamicGPUManager, interval: int = 30) -> GPUMonitor:
    """Start GPU monitoring service"""
    monitor = GPUMonitor(gpu_manager, interval)
    monitor.start_monitoring()
    return monitor

# =====================================
# MAIN EXPORT CLASS - Simplified Interface
# =====================================

class ImprovedDynamicGPUManager(DynamicGPUManager):
    """Main export class with simplified interface for backward compatibility"""
    
    def __init__(self, total_gpu_count: int = 4, memory_threshold: float = 0.80, 
                 utilization_threshold: float = 75.0):
        super().__init__(total_gpu_count, memory_threshold, utilization_threshold)
        
        # Additional convenience methods for backward compatibility
        self.recovery_manager = GPURecoveryManager()
        self.monitor = None
    
    def start_monitoring(self, interval: int = 30):
        """Start GPU monitoring"""
        if self.monitor is None:
            self.monitor = GPUMonitor(self, interval)
            self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if self.monitor:
            self.monitor.stop_monitoring()
            self.monitor = None
    
    def emergency_recovery(self) -> Dict[str, Any]:
        """Perform emergency GPU recovery"""
        return self.recovery_manager.recover_from_gpu_errors()
    
    def get_available_gpu(self, preferred_gpu: Optional[int] = None, 
                         fallback: bool = True, allow_sharing: bool = False) -> Optional[int]:
        """Backward compatibility method"""
        return self.get_available_gpu_smart(
            preferred_gpu=preferred_gpu,
            workload_type="general",
            allow_sharing=allow_sharing,
            exclude_gpu_0=True
        )

# =====================================
# CONVENIENCE FUNCTIONS
# =====================================

def create_gpu_manager(total_gpus: int = 4) -> ImprovedDynamicGPUManager:
    """Create a GPU manager with default settings"""
    return ImprovedDynamicGPUManager(total_gpu_count=total_gpus)

def check_all_gpus() -> Dict[int, Dict[str, Any]]:
    """Quick check of all GPU statuses"""
    results = {}
    for gpu_id in range(4):
        results[gpu_id] = GPUHardwareInterface.check_gpu_memory(gpu_id)
    return results

def find_best_gpu(min_memory_gb: float = 2.0) -> Optional[int]:
    """Find the best available GPU"""
    return GPUAllocationStrategy.find_best_gpu_with_memory(min_memory_gb)

def cleanup_gpu(gpu_id: int) -> bool:
    """Clean up a specific GPU"""
    return GPUHardwareInterface.aggressive_gpu_cleanup(gpu_id)

def force_gpu_environment(gpu_id: int) -> bool:
    """Force GPU environment for specific GPU"""
    return GPUEnvironmentManager.safe_force_gpu_environment(gpu_id)

if __name__ == "__main__":
    # Run tests if script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_gpu_manager()
    
    # Test recovery
    print("\n🔧 Testing GPU Recovery:")
    recovery_result = GPURecoveryManager.recover_from_gpu_errors()
    print(f"Recovery completed: {recovery_result['status']}")
    print(f"Success rate: {recovery_result.get('success_rate', 0):.1%}")