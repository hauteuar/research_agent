# utils/gpu_manager.py - OPTIMIZED VERSION
import torch
import psutil
import subprocess
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime as dt
import functools

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU configuration with auto-detection"""
    total_gpu_count: int = 0  # Will be auto-detected
    memory_threshold: float = 0.7
    utilization_threshold: float = 70.0
    min_memory_gb: float = 2.0
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems

class GPUStatus:
    """GPU status tracking"""
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.is_available = False
        self.memory_free_gb = 0.0
        self.memory_total_gb = 0.0
        self.utilization_percent = 0.0
        self.temperature = 0.0
        self.process_count = 0
        self.active_workloads = []
        self.last_updated = dt.now()
        self.error_count = 0

class OptimizedDynamicGPUManager:
    """Optimized GPU Manager with proper GPU detection and shared system safety"""
    
    def __init__(self, total_gpu_count: Optional[int] = None, 
                 memory_threshold: float = 0.7, 
                 utilization_threshold: float = 70.0):
        """Initialize with auto-detection and safety for shared systems"""
        
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        
        # Auto-detect GPU count if not provided
        self.total_gpu_count = self._detect_gpu_count() if total_gpu_count is None else total_gpu_count
        self.available_gpu_count = self._get_available_gpu_count()
        
        # GPU status tracking
        self.gpu_status: Dict[int, GPUStatus] = {}
        self._status_lock = threading.Lock()
        self._last_refresh = 0
        self._refresh_interval = 30  # seconds
        
        # Workload tracking
        self.active_workloads: Dict[int, List[str]] = {}
        self.workload_history: List[Dict] = []
        
        # Initialize GPU status
        self._initialize_gpu_status()
        
        logger.info(f"✅ Optimized GPU Manager: {self.available_gpu_count}/{self.total_gpu_count} GPUs available")
    
    @functools.lru_cache(maxsize=1)
    def _detect_gpu_count(self) -> int:
        """Auto-detect total GPU count with caching"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return 0
            
            gpu_count = torch.cuda.device_count()
            logger.info(f"🔍 Detected {gpu_count} GPUs")
            return gpu_count
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return 0
    
    def _get_available_gpu_count(self) -> int:
        """Get count of actually usable GPUs"""
        if self.total_gpu_count == 0:
            return 0
        
        available = 0
        start_gpu = 1 if self.total_gpu_count > 1 else 0  # Skip GPU 0 if multiple GPUs
        
        for gpu_id in range(start_gpu, self.total_gpu_count):
            if self._is_gpu_usable(gpu_id):
                available += 1
        
        return available
    
    def _is_gpu_usable(self, gpu_id: int) -> bool:
        """Check if GPU is usable (quick check)"""
        try:
            # Quick memory check
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                free_mb, util = map(int, result.stdout.strip().split(','))
                free_gb = free_mb / 1024
                
                return (free_gb >= 2.0 and util < 90)  # Basic usability check
                
        except Exception as e:
            logger.warning(f"GPU {gpu_id} usability check failed: {e}")
        
        return False
    
    def _initialize_gpu_status(self):
        """Initialize status tracking for available GPUs"""
        start_gpu = 1 if self.total_gpu_count > 1 else 0
        
        for gpu_id in range(start_gpu, self.total_gpu_count):
            self.gpu_status[gpu_id] = GPUStatus(gpu_id)
            self.active_workloads[gpu_id] = []
        
        # Do initial status refresh
        self._refresh_gpu_status()
    
    def _refresh_gpu_status(self, force: bool = False):
        """Refresh GPU status (with rate limiting)"""
        current_time = time.time()
        
        if not force and (current_time - self._last_refresh) < self._refresh_interval:
            return
        
        with self._status_lock:
            for gpu_id in self.gpu_status.keys():
                try:
                    self._update_gpu_status(gpu_id)
                except Exception as e:
                    logger.warning(f"Status update failed for GPU {gpu_id}: {e}")
                    self.gpu_status[gpu_id].error_count += 1
            
            self._last_refresh = current_time
    
    def _update_gpu_status(self, gpu_id: int):
        """Update status for specific GPU"""
        try:
            # Get GPU info
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                total, used, free, util, temp = map(int, result.stdout.strip().split(','))
                
                status = self.gpu_status[gpu_id]
                status.memory_total_gb = total / 1024
                status.memory_free_gb = free / 1024
                status.utilization_percent = util
                status.temperature = temp
                status.last_updated = dt.now()
                
                # Determine availability
                status.is_available = (
                    status.memory_free_gb >= 2.0 and
                    status.utilization_percent < self.utilization_threshold and
                    status.temperature < 85 and
                    len(self.active_workloads[gpu_id]) < 3  # Max 3 concurrent workloads
                )
                
                # Get process count
                proc_result = subprocess.run([
                    'nvidia-smi', f'--id={gpu_id}',
                    '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if proc_result.returncode == 0:
                    process_lines = [line.strip() for line in proc_result.stdout.strip().split('\n') if line.strip()]
                    status.process_count = len(process_lines)
                
        except Exception as e:
            logger.warning(f"Failed to update GPU {gpu_id} status: {e}")
            # Mark as unavailable on error
            self.gpu_status[gpu_id].is_available = False
    
    def get_optimal_gpu(self, workload_type: str = "default", 
                       min_memory_gb: float = 2.0) -> Optional[int]:
        """Get optimal GPU for workload with intelligent selection"""
        
        # Refresh status if needed
        self._refresh_gpu_status()
        
        candidates = []
        
        with self._status_lock:
            for gpu_id, status in self.gpu_status.items():
                if not status.is_available:
                    continue
                
                if status.memory_free_gb < min_memory_gb:
                    continue
                
                # Calculate score
                score = self._calculate_gpu_score(status, workload_type)
                candidates.append((gpu_id, score, status))
        
        if not candidates:
            logger.warning("No suitable GPU found")
            return None
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_gpu = candidates[0][0]
        best_score = candidates[0][1]
        
        logger.info(f"Selected GPU {best_gpu} (score: {best_score:.2f}) for {workload_type}")
        return best_gpu
    
    def _calculate_gpu_score(self, status: GPUStatus, workload_type: str) -> float:
        """Calculate GPU suitability score"""
        score = 0.0
        
        # Free memory weight (most important)
        score += status.memory_free_gb * 20
        
        # Lower utilization is better
        score += (100 - status.utilization_percent) * 0.5
        
        # Lower temperature is better
        score += (100 - status.temperature) * 0.2
        
        # Fewer active workloads is better
        workload_count = len(self.active_workloads[status.gpu_id])
        score += (5 - workload_count) * 10
        
        # Fewer processes is better
        score += (20 - status.process_count) * 2
        
        # Bonus for GPU 1,2,3 over GPU 0 in shared systems
        if status.gpu_id > 0:
            score += 10
        
        return max(score, 0.0)
    
    def reserve_gpu_for_workload(self, workload_type: str, 
                                preferred_gpu: Optional[int] = None,
                                duration_estimate: int = 300,
                                allow_sharing: bool = True) -> Optional[int]:
        """Reserve GPU for workload with proper tracking"""
        
        # Try preferred GPU first
        if preferred_gpu is not None:
            if self._can_reserve_gpu(preferred_gpu, allow_sharing):
                return self._reserve_gpu(preferred_gpu, workload_type, duration_estimate)
        
        # Find optimal GPU
        optimal_gpu = self.get_optimal_gpu(workload_type)
        if optimal_gpu is not None:
            if self._can_reserve_gpu(optimal_gpu, allow_sharing):
                return self._reserve_gpu(optimal_gpu, workload_type, duration_estimate)
        
        logger.warning(f"Could not reserve GPU for {workload_type}")
        return None
    
    def _can_reserve_gpu(self, gpu_id: int, allow_sharing: bool) -> bool:
        """Check if GPU can be reserved"""
        if gpu_id not in self.gpu_status:
            return False
        
        status = self.gpu_status[gpu_id]
        
        if not status.is_available:
            return False
        
        workload_count = len(self.active_workloads[gpu_id])
        
        if not allow_sharing and workload_count > 0:
            return False
        
        if workload_count >= 3:  # Hard limit
            return False
        
        return True
    
    def _reserve_gpu(self, gpu_id: int, workload_type: str, duration: int) -> int:
        """Actually reserve the GPU"""
        workload_id = f"{workload_type}_{int(time.time())}"
        
        self.active_workloads[gpu_id].append({
            'workload_id': workload_id,
            'workload_type': workload_type,
            'start_time': time.time(),
            'estimated_duration': duration
        })
        
        logger.info(f"Reserved GPU {gpu_id} for {workload_type} (workload_id: {workload_id})")
        return gpu_id
    
    def release_gpu_workload(self, gpu_id: int, workload_identifier: str):
        """Release GPU workload"""
        if gpu_id not in self.active_workloads:
            return
        
        # Remove workload by identifier (can be workload_type or workload_id)
        original_count = len(self.active_workloads[gpu_id])
        
        self.active_workloads[gpu_id] = [
            w for w in self.active_workloads[gpu_id]
            if not (workload_identifier in w['workload_id'] or 
                   workload_identifier in w['workload_type'])
        ]
        
        released_count = original_count - len(self.active_workloads[gpu_id])
        
        if released_count > 0:
            logger.info(f"Released {released_count} workload(s) from GPU {gpu_id}")
    
    def cleanup_completed_workloads(self):
        """Clean up workloads that should have completed"""
        current_time = time.time()
        
        for gpu_id in self.active_workloads:
            original_count = len(self.active_workloads[gpu_id])
            
            # Remove workloads that have exceeded their estimated duration by 2x
            self.active_workloads[gpu_id] = [
                w for w in self.active_workloads[gpu_id]
                if (current_time - w['start_time']) < (w['estimated_duration'] * 2)
            ]
            
            cleaned = original_count - len(self.active_workloads[gpu_id])
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} expired workloads from GPU {gpu_id}")
    
    def get_gpu_status_detailed(self) -> Dict[str, Any]:
        """Get detailed status of all GPUs"""
        self._refresh_gpu_status()
        
        status_dict = {}
        
        with self._status_lock:
            for gpu_id, status in self.gpu_status.items():
                status_dict[f"gpu_{gpu_id}"] = {
                    "gpu_id": gpu_id,
                    "is_available": status.is_available,
                    "memory_free_gb": status.memory_free_gb,
                    "memory_total_gb": status.memory_total_gb,
                    "utilization_percent": status.utilization_percent,
                    "temperature": status.temperature,
                    "process_count": status.process_count,
                    "active_workloads": len(self.active_workloads[gpu_id]),
                    "workload_details": self.active_workloads[gpu_id],
                    "last_updated": status.last_updated.isoformat(),
                    "error_count": status.error_count
                }
        
        return status_dict
    
    def get_system_gpu_summary(self) -> Dict[str, Any]:
        """Get high-level GPU system summary"""
        self._refresh_gpu_status()
        
        available_gpus = sum(1 for status in self.gpu_status.values() if status.is_available)
        total_workloads = sum(len(workloads) for workloads in self.active_workloads.values())
        
        return {
            "total_gpus": self.total_gpu_count,
            "available_gpus": available_gpus,
            "managed_gpus": len(self.gpu_status),
            "total_active_workloads": total_workloads,
            "gpu_utilization_avg": sum(s.utilization_percent for s in self.gpu_status.values()) / len(self.gpu_status) if self.gpu_status else 0,
            "memory_free_total_gb": sum(s.memory_free_gb for s in self.gpu_status.values()),
            "system_health": "healthy" if available_gpus > 0 else "no_gpus_available"
        }
    
    def force_refresh(self):
        """Force immediate status refresh"""
        self._refresh_gpu_status(force=True)
    
    def get_recommendation(self, workload_type: str) -> Dict[str, Any]:
        """Get GPU recommendation for workload type"""
        optimal_gpu = self.get_optimal_gpu(workload_type)
        
        if optimal_gpu is None:
            return {
                "recommended_gpu": None,
                "reason": "No suitable GPU available",
                "alternatives": [],
                "wait_suggested": True
            }
        
        status = self.gpu_status[optimal_gpu]
        
        return {
            "recommended_gpu": optimal_gpu,
            "reason": f"Best available: {status.memory_free_gb:.1f}GB free, {status.utilization_percent}% util",
            "memory_available": status.memory_free_gb,
            "current_utilization": status.utilization_percent,
            "active_workloads": len(self.active_workloads[optimal_gpu]),
            "estimated_wait_time": 0
        }
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down GPU manager...")
        
        # Clear all workloads
        for gpu_id in self.active_workloads:
            self.active_workloads[gpu_id].clear()
        
        logger.info("GPU manager shutdown complete")
    
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

# For backward compatibility
DynamicGPUManager = OptimizedDynamicGPUManager