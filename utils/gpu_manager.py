# Updated GPU Manager with Enhanced Error Handling and Recovery

import torch
import psutil
import logging
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

class EnhancedGPUForcer:
    """Enhanced GPU forcing with better error handling and memory management"""
    
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
                        
                        # Get process information with error handling
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
                            'can_share': free > 2048 and len(processes) < 3,  # Can accept more workloads
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
    def find_best_gpu_with_memory(min_free_gb: float = 2.0, exclude_gpu_0: bool = True, 
                                 allow_sharing: bool = True, prioritize_health: bool = True) -> Optional[int]:
        """Enhanced GPU finding with health considerations and sharing support"""
        candidates = []
        gpu_info = {}
        
        # Check all GPUs
        gpu_range = range(1 if exclude_gpu_0 else 0, 4)
        
        for gpu_id in gpu_range:
            memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            free_gb = memory_info['free_gb']
            gpu_info[gpu_id] = memory_info
            
            logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free, {memory_info['gpu_utilization']:.1f}% util, "
                       f"temp: {memory_info['temperature']:.1f}°C, processes: {len(memory_info['processes'])}, "
                       f"healthy: {memory_info['is_healthy']}, can_share: {memory_info['can_share']}")
            
            # Calculate suitability score
            score = 0
            
            if memory_info['is_available']:
                score += 100
                
                # Memory score
                if free_gb >= min_free_gb:
                    score += min(free_gb * 10, 100)  # Up to 100 points for memory
                
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
                
                candidates.append((gpu_id, score, free_gb, memory_info))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Log all candidates
        for gpu_id, score, free_gb, info in candidates:
            logger.info(f"GPU {gpu_id}: score={score:.1f}, free={free_gb:.1f}GB, "
                       f"healthy={info['is_healthy']}, can_share={info['can_share']}")
        
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
            return EnhancedGPUForcer.find_best_gpu_with_memory(
                min_free_gb=max(1.0, min_free_gb * 0.5),
                exclude_gpu_0=exclude_gpu_0,
                allow_sharing=True,
                prioritize_health=False
            )
        
        # Last resort: include GPU 0 if we were excluding it
        if exclude_gpu_0:
            logger.warning("No suitable GPU found excluding GPU 0, checking GPU 0 as last resort...")
            memory_info = EnhancedGPUForcer.check_gpu_memory(0)
            gpu_info[0] = memory_info
            
            if memory_info['free_gb'] >= max(0.5, min_free_gb * 0.25):  # Very relaxed requirement
                logger.warning(f"Using GPU 0 as last resort with {memory_info['free_gb']:.1f}GB free")
                return 0
        
        logger.error(f"No GPU found with at least {min_free_gb}GB free")
        logger.error("GPU Status Summary:")
        for gpu_id, info in gpu_info.items():
            logger.error(f"  GPU {gpu_id}: {info['free_gb']:.1f}GB free, {len(info['processes'])} processes, "
                        f"healthy: {info.get('is_healthy', False)}, error: {info.get('error', 'none')}")
        
        return None
    
    @staticmethod
    def aggressive_gpu_cleanup(gpu_id: int) -> bool:
        """Perform aggressive GPU cleanup with multiple strategies"""
        logger.info(f"Starting aggressive cleanup for GPU {gpu_id}")
        
        try:
            # Strategy 1: Python garbage collection
            logger.info("Step 1: Python garbage collection")
            gc.collect()
            
            # Strategy 2: PyTorch cleanup
            if torch.cuda.is_available():
                logger.info("Step 2: PyTorch CUDA cleanup")
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(gpu_id)
                        torch.cuda.reset_accumulated_memory_stats(gpu_id)
                except Exception as e:
                    logger.warning(f"PyTorch cleanup failed: {e}")
            
            # Strategy 3: Force garbage collection again
            logger.info("Step 3: Second garbage collection")
            import gc
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Strategy 4: Check and wait for cleanup to take effect
            logger.info("Step 4: Verifying cleanup results")
            initial_memory = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            
            # Wait a bit for cleanup to take effect
            time.sleep(2)
            
            final_memory = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            
            freed_mb = final_memory['free_mb'] - initial_memory['free_mb']
            logger.info(f"Cleanup freed {freed_mb}MB on GPU {gpu_id}")
            
            return final_memory['free_gb'] > initial_memory['free_gb']
            
        except Exception as e:
            logger.error(f"Aggressive cleanup failed for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def safe_force_gpu_environment(gpu_id: int, cleanup_first: bool = True, 
                                  verify_success: bool = True) -> bool:
        """Safely force GPU environment with verification"""
        try:
            if cleanup_first:
                logger.info(f"Pre-cleanup for GPU {gpu_id}")
                EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
            
            # Check GPU availability before forcing
            memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            if not memory_info['is_available']:
                logger.warning(f"GPU {gpu_id} is not available before forcing environment")
                if not cleanup_first:  # Try cleanup if we haven't already
                    EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    
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
        from vllm import AsyncEngineArgs
        
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

class ImprovedDynamicGPUManager(DynamicGPUManager):
    """Improved GPU Manager with enhanced error recovery and request management"""
    
    def __init__(self, total_gpu_count: int = 4, memory_threshold: float = 0.80, 
                 utilization_threshold: float = 75.0):
        super().__init__(total_gpu_count, memory_threshold, utilization_threshold)
        
        # Enhanced error tracking
        self.gpu_error_counts = defaultdict(int)
        self.gpu_last_errors = {}
        self.recovery_attempts = defaultdict(int)
        
        # Request management to prevent flooding
        self.request_semaphores = {}
        self.active_requests = defaultdict(int)
        
        # Initialize request semaphores for each GPU
        for gpu_id in range(total_gpu_count):
            self.request_semaphores[gpu_id] = threading.Semaphore(2)  # Max 2 concurrent requests per GPU
    
    def _get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Enhanced GPU info with error tracking"""
        try:
            # Use enhanced GPU forcer
            info_dict = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            
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
            
            # Determine status with error consideration
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
        """Smart GPU allocation with workload awareness and error recovery"""
        
        with self.lock:
            # Force refresh with error handling
            try:
                self._update_all_gpu_info()
            except Exception as e:
                logger.error(f"Failed to update GPU info: {e}")
                return None
            
            # Try enhanced GPU forcer first
            try:
                # Determine memory requirements based on workload type
                if "llm_engine" in workload_type.lower():
                    min_memory = 3.0  # LLM engines need more memory
                    allow_sharing = False  # LLM engines prefer exclusive access
                elif "embedding" in workload_type.lower():
                    min_memory = 1.5  # Embedding models need moderate memory
                elif "analysis" in workload_type.lower():
                    min_memory = 1.0  # Analysis can work with less
                else:
                    min_memory = 2.0  # Default requirement
                
                best_gpu = EnhancedGPUForcer.find_best_gpu_with_memory(
                    min_free_gb=min_memory,
                    exclude_gpu_0=exclude_gpu_0,
                    allow_sharing=allow_sharing,
                    prioritize_health=True
                )
                
                if best_gpu is not None:
                    logger.info(f"Enhanced GPU forcer selected GPU {best_gpu} for {workload_type}")
                    return best_gpu
                    
            except Exception as e:
                logger.warning(f"Enhanced GPU forcer failed: {e}, falling back to standard method")
            
            # Fallback to standard method
            return super().get_available_gpu(preferred_gpu, fallback=True, allow_sharing=allow_sharing)
    
    def attempt_gpu_recovery(self, gpu_id: int) -> bool:
        """Attempt to recover a problematic GPU"""
        if self.recovery_attempts[gpu_id] >= 3:
            logger.error(f"GPU {gpu_id} recovery limit reached, marking as permanently failed")
            return False
        
        logger.info(f"Attempting recovery for GPU {gpu_id} (attempt {self.recovery_attempts[gpu_id] + 1})")
        self.recovery_attempts[gpu_id] += 1
        
        try:
            # Step 1: Aggressive cleanup
            success = EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
            if not success:
                logger.warning(f"Cleanup failed for GPU {gpu_id}")
            
            # Step 2: Wait for recovery
            time.sleep(5)
            
            # Step 3: Test GPU
            memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            if memory_info['is_available']:
                logger.info(f"GPU {gpu_id} recovery successful")
                self.recovery_attempts[gpu_id] = 0  # Reset on successful recovery
                return True
            else:
                logger.warning(f"GPU {gpu_id} recovery failed - still not available")
                return False
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} recovery attempt failed: {e}")
            return False
    
    def get_detailed_gpu_report(self) -> Dict[str, Any]:
        """Get comprehensive GPU status report"""
        with self.lock:
            report = {
                "timestamp": time.time(),
                "gpus": {},
                "summary": {
                    "total_gpus": self.total_gpu_count,
                    "available_gpus": 0,
                    "busy_gpus": 0,
                    "error_gpus": 0,
                    "recovering_gpus": 0
                },
                "workload_distribution": self.get_workload_distribution(),
                "sharing_info": self.get_gpu_sharing_info()
            }
            
            for gpu_id, info in self.gpu_info.items():
                report["gpus"][f"gpu_{gpu_id}"] = {
                    "id": gpu_id,
                    "name": info.name,
                    "status": info.status.value,
                    "memory_total_gb": round(info.total_memory / (1024**3), 2),
                    "memory_free_gb": round(info.free_memory / (1024**3), 2),
                    "memory_utilization_percent": round((info.used_memory / max(info.total_memory, 1)) * 100, 2),
                    "gpu_utilization_percent": round(info.utilization, 2),
                    "temperature_celsius": round(info.temperature, 1),
                    "process_count": info.process_count,
                    "error_count": info.error_count,
                    "last_error": info.last_error,
                    "last_updated": info.last_updated,
                    "recovery_attempts": self.recovery_attempts.get(gpu_id, 0),
                    "active_requests": self.active_requests.get(gpu_id, 0)
                }
                
                # Update summary
                if info.status == GPUStatus.AVAILABLE:
                    report["summary"]["available_gpus"] += 1
                elif info.status == GPUStatus.BUSY:
                    report["summary"]["busy_gpus"] += 1
                elif info.status == GPUStatus.ERROR:
                    report["summary"]["error_gpus"] += 1
                elif info.status == GPUStatus.RECOVERING:
                    report["summary"]["recovering_gpus"] += 1
            
            return report
    
    def safe_request_gpu(self, gpu_id: int, workload_type: str, timeout: float = 30.0) -> bool:
        """Safely request GPU access with timeout and rate limiting"""
        try:
            # Get semaphore for this GPU
            semaphore = self.request_semaphores.get(gpu_id)
            if semaphore is None:
                logger.error(f"No semaphore found for GPU {gpu_id}")
                return False
            
            # Try to acquire semaphore with timeout
            acquired = semaphore.acquire(timeout=timeout)
            if not acquired:
                logger.warning(f"Timeout waiting for GPU {gpu_id} access")
                return False
            
            try:
                # Track active request
                self.active_requests[gpu_id] += 1
                logger.info(f"Acquired GPU {gpu_id} for {workload_type} "
                           f"({self.active_requests[gpu_id]} active requests)")
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
                
            logger.info(f"Released GPU {gpu_id} from {workload_type} "
                       f"({self.active_requests[gpu_id]} remaining requests)")
                       
        except Exception as e:
            logger.error(f"Error releasing GPU {gpu_id}: {e}")

# Context manager for safe GPU usage
class SafeGPUContext:
    """Context manager for safe GPU allocation and cleanup"""
    
    def __init__(self, gpu_manager: ImprovedDynamicGPUManager, workload_type: str,
                 preferred_gpu: Optional[int] = None, cleanup_on_exit: bool = True):
        self.gpu_manager = gpu_manager
        self.workload_type = workload_type
        self.preferred_gpu = preferred_gpu
        self.cleanup_on_exit = cleanup_on_exit
        self.allocated_gpu = None
        self.request_acquired = False
    
    def __enter__(self) -> Optional[int]:
        try:
            # Get GPU allocation
            self.allocated_gpu = self.gpu_manager.get_available_gpu_smart(
                preferred_gpu=self.preferred_gpu,
                workload_type=self.workload_type,
                allow_sharing="llm_engine" not in self.workload_type.lower()
            )
            
            if self.allocated_gpu is None:
                logger.error(f"No GPU available for {self.workload_type}")
                return None
            
            # Request safe access
            self.request_acquired = self.gpu_manager.safe_request_gpu(
                self.allocated_gpu, self.workload_type
            )
            
            if not self.request_acquired:
                logger.error(f"Failed to acquire safe access to GPU {self.allocated_gpu}")
                self.allocated_gpu = None
                return None
            
            # Force GPU environment
            success = EnhancedGPUForcer.safe_force_gpu_environment(
                self.allocated_gpu, 
                cleanup_first=True,
                verify_success=True
            )
            
            if not success:
                logger.error(f"Failed to force GPU environment for GPU {self.allocated_gpu}")
                self.gpu_manager.release_gpu_request(self.allocated_gpu, self.workload_type)
                self.allocated_gpu = None
                self.request_acquired = False
                return None
            
            logger.info(f"Successfully allocated GPU {self.allocated_gpu} for {self.workload_type}")
            return self.allocated_gpu
            
        except Exception as e:
            logger.error(f"GPU context setup failed: {e}")
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.allocated_gpu is not None:
                if self.cleanup_on_exit:
                    logger.info(f"Cleaning up GPU {self.allocated_gpu}")
                    EnhancedGPUForcer.aggressive_gpu_cleanup(self.allocated_gpu)
                
                if self.request_acquired:
                    self.gpu_manager.release_gpu_request(self.allocated_gpu, self.workload_type)
                
                logger.info(f"Released GPU {self.allocated_gpu} from {self.workload_type}")
                
        except Exception as e:
            logger.error(f"GPU context cleanup failed: {e}")

# Usage example and testing functions
def test_enhanced_gpu_manager():
    """Test the enhanced GPU manager"""
    print("Testing Enhanced GPU Manager")
    print("=" * 50)
    
    # Create manager
    manager = ImprovedDynamicGPUManager()
    
    # Wait for initial GPU scan
    time.sleep(2)
    
    # Get detailed report
    report = manager.get_detailed_gpu_report()
    
    print("GPU Status Report:")
    for gpu_name, gpu_info in report["gpus"].items():
        print(f"{gpu_name}: {gpu_info['status']}, "
              f"{gpu_info['memory_free_gb']:.1f}GB free, "
              f"{gpu_info['gpu_utilization_percent']:.1f}% util, "
              f"errors: {gpu_info['error_count']}")
    
    print(f"\nSummary: {report['summary']['available_gpus']} available, "
          f"{report['summary']['busy_gpus']} busy, "
          f"{report['summary']['error_gpus']} error")
    
    # Test GPU allocation
    print("\nTesting GPU allocation...")
    
    # Test with context manager
    with SafeGPUContext(manager, "test_workload") as gpu_id:
        if gpu_id is not None:
            print(f"✅ Successfully allocated GPU {gpu_id}")
            
            # Test memory check
            memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
            print(f"GPU {gpu_id} memory: {memory_info['free_gb']:.1f}GB free")
        else:
            print("❌ Failed to allocate GPU")
    
    # Cleanup
    manager.shutdown()
    print("✅ GPU manager test completed")

# Integration functions for the coordinator
async def integrate_with_coordinator(coordinator):
    """Integrate enhanced GPU manager with coordinator"""
    
    # Replace coordinator's GPU manager
    coordinator.gpu_manager = ImprovedDynamicGPUManager()
    
    # Update coordinator's get_available_gpu_for_agent method
    async def enhanced_get_available_gpu_for_agent(self, agent_type: str, 
                                                  preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Enhanced GPU allocation for agents"""
        
        # Force refresh GPU status
        self.gpu_manager.force_refresh()
        
        # Check for existing engines that can be shared
        existing_engines = list(self.llm_engine_pool.keys())
        if existing_engines and "llm" not in agent_type.lower():
            for engine_key in existing_engines:
                gpu_id = int(engine_key.split('_')[1])
                
                try:
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    if memory_info.get('can_share', False):
                        self.logger.info(f"Sharing existing LLM engine on GPU {gpu_id} for {agent_type}")
                        return gpu_id
                except Exception as e:
                    self.logger.warning(f"Error checking GPU {gpu_id} for sharing: {e}")
                    continue
        
        # Use smart GPU allocation
        best_gpu = self.gpu_manager.get_available_gpu_smart(
            preferred_gpu=preferred_gpu,
            workload_type=agent_type,
            allow_sharing="llm_engine" not in agent_type.lower(),
            exclude_gpu_0=True  # Prefer to avoid GPU 0
        )
        
        if best_gpu is not None:
            self.logger.info(f"Allocated GPU {best_gpu} for {agent_type}")
            return best_gpu
        
        # Last resort: try any GPU including GPU 0
        best_gpu = self.gpu_manager.get_available_gpu_smart(
            preferred_gpu=None,
            workload_type=agent_type,
            allow_sharing=True,
            exclude_gpu_0=False
        )
        
        if best_gpu is not None:
            self.logger.warning(f"Using GPU {best_gpu} for {agent_type} as last resort")
            return best_gpu
        
        self.logger.error(f"No GPU available for {agent_type}")
        return None
    
    # Replace method on coordinator instance
    coordinator.get_available_gpu_for_agent = enhanced_get_available_gpu_for_agent.__get__(coordinator)
    
    # Update LLM engine creation method
    async def enhanced_get_or_create_llm_engine(self, gpu_id: int, force_reload: bool = False):
        """Enhanced LLM engine creation with improved memory management"""
        async with self.engine_lock:
            engine_key = f"gpu_{gpu_id}"
            
            # Return existing engine if available and not forcing reload
            if engine_key in self.llm_engine_pool and not force_reload:
                self.logger.info(f"Reusing existing LLM engine on GPU {gpu_id}")
                return self.llm_engine_pool[engine_key]
            
            # Use safe GPU context for engine creation
            with SafeGPUContext(self.gpu_manager, f"llm_engine_{engine_key}", 
                              preferred_gpu=gpu_id, cleanup_on_exit=False) as allocated_gpu:
                
                if allocated_gpu != gpu_id:
                    if allocated_gpu is None:
                        raise RuntimeError(f"Failed to allocate GPU {gpu_id} for LLM engine")
                    else:
                        self.logger.warning(f"GPU {gpu_id} not available, using GPU {allocated_gpu} instead")
                        gpu_id = allocated_gpu
                        engine_key = f"gpu_{gpu_id}"
                
                try:
                    # Check final memory state
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info['free_gb']
                    
                    if free_gb < 1.5:
                        raise RuntimeError(f"Insufficient memory on GPU {gpu_id}: {free_gb:.1f}GB free")
                    
                    self.logger.info(f"Creating LLM engine on GPU {gpu_id} with {free_gb:.1f}GB available")
                    
                    # Create conservative engine args
                    engine_args = EnhancedGPUForcer.create_conservative_vllm_args(
                        self.config.model_name,
                        self.config.max_tokens
                    )
                    
                    # Create engine
                    from vllm import AsyncLLMEngine
                    engine = AsyncLLMEngine.from_engine_args(engine_args)
                    
                    self.llm_engine_pool[engine_key] = engine
                    
                    # Reserve GPU in manager
                    self.gpu_manager.reserve_gpu_for_workload(
                        workload_type=f"llm_engine_{engine_key}",
                        preferred_gpu=gpu_id,
                        duration_estimate=3600,
                        allow_sharing=True
                    )
                    
                    # Verify final state
                    final_memory = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    used_gb = free_gb - final_memory['free_gb']
                    
                    self.logger.info(f"✅ LLM engine created on GPU {gpu_id}. "
                                   f"Used {used_gb:.1f}GB, {final_memory['free_gb']:.1f}GB remaining")
                    
                    return engine
                    
                except Exception as e:
                    self.logger.error(f"Failed to create LLM engine on GPU {gpu_id}: {e}")
                    
                    # Cleanup on failure
                    if engine_key in self.llm_engine_pool:
                        del self.llm_engine_pool[engine_key]
                    
                    # Release GPU workload
                    try:
                        self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
                    except:
                        pass
                    
                    # Perform cleanup
                    EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                    
                    raise
    
    # Replace method on coordinator instance
    coordinator.get_or_create_llm_engine = enhanced_get_or_create_llm_engine.__get__(coordinator)
    
    coordinator.logger.info("Enhanced GPU manager integrated with coordinator")

# Error recovery functions
def recover_from_gpu_errors():
    """Recover from common GPU errors"""
    logger.info("Starting GPU error recovery process...")
    
    try:
        # Step 1: Check all GPUs
        logger.info("Step 1: Checking all GPU states")
        gpu_states = {}
        
        for gpu_id in range(4):
            try:
                memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                gpu_states[gpu_id] = memory_info
                logger.info(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free, "
                           f"healthy: {memory_info.get('is_healthy', False)}")
            except Exception as e:
                logger.error(f"GPU {gpu_id} check failed: {e}")
                gpu_states[gpu_id] = {'error': str(e)}
        
        # Step 2: Cleanup problematic GPUs
        logger.info("Step 2: Cleaning up problematic GPUs")
        for gpu_id, state in gpu_states.items():
            if 'error' in state or not state.get('is_healthy', False):
                logger.info(f"Attempting recovery for GPU {gpu_id}")
                success = EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                if success:
                    logger.info(f"✅ GPU {gpu_id} recovery successful")
                else:
                    logger.warning(f"❌ GPU {gpu_id} recovery failed")
        
        # Step 3: Verify recovery
        logger.info("Step 3: Verifying recovery")
        recovered_count = 0
        for gpu_id in range(4):
            try:
                memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                if memory_info.get('is_available', False):
                    recovered_count += 1
                    logger.info(f"✅ GPU {gpu_id} is now available")
                else:
                    logger.warning(f"❌ GPU {gpu_id} still not available")
            except Exception as e:
                logger.error(f"GPU {gpu_id} verification failed: {e}")
        
        logger.info(f"Recovery complete: {recovered_count}/4 GPUs available")
        return recovered_count > 0
        
    except Exception as e:
        logger.error(f"GPU recovery process failed: {e}")
        return False

# Monitoring functions
def start_gpu_monitoring(interval: int = 30):
    """Start continuous GPU monitoring"""
    import threading
    
    def monitor_loop():
        while True:
            try:
                logger.info("=== GPU Monitoring Report ===")
                
                for gpu_id in range(4):
                    try:
                        memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                        logger.info(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free, "
                                   f"{memory_info['gpu_utilization']:.1f}% util, "
                                   f"{memory_info['temperature']:.1f}°C, "
                                   f"processes: {len(memory_info['processes'])}")
                    except Exception as e:
                        logger.error(f"GPU {gpu_id} monitoring failed: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="GPU_Monitor")
    monitor_thread.start()
    logger.info(f"Started GPU monitoring with {interval}s interval")

if __name__ == "__main__":
    # Test the enhanced GPU manager
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Enhanced GPU Manager")
    test_enhanced_gpu_manager()
    
    print("\n🔧 Testing GPU Error Recovery")
    recover_from_gpu_errors()
    
    print("\n📊 Starting GPU Monitoring (Ctrl+C to stop)")
    try:
        start_gpu_monitoring(10)
        time.sleep(60)  # Monitor for 1 minute
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")