# utils/gpu_manager.py - COMPLETE VERSION WITH ALL MISSING COMPONENTS
import torch
import psutil
import subprocess
import time
import logging
import threading
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime as dt, timedelta
import functools
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    """Workload types for better GPU allocation"""
    LLM_ENGINE = "llm_engine"
    VECTOR_INDEX = "vector_index"
    CODE_PARSER = "code_parser"
    DATA_LOADER = "data_loader"
    LINEAGE_ANALYZER = "lineage_analyzer"
    LOGIC_ANALYZER = "logic_analyzer"
    DOCUMENTATION = "documentation"
    DB2_COMPARATOR = "db2_comparator"
    CHAT_AGENT = "chat_agent"
    BATCH_PROCESSING = "batch_processing"
    UNKNOWN = "unknown"

@dataclass
class GPUConfig:
    """GPU configuration with auto-detection"""
    total_gpu_count: int = 0  # Will be auto-detected
    memory_threshold: float = 0.7
    utilization_threshold: float = 70.0
    min_memory_gb: float = 2.0
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems
    max_workloads_per_gpu: int = 3
    workload_timeout_multiplier: float = 2.0
    temperature_threshold: float = 85.0
    refresh_interval: int = 30  # seconds

@dataclass
class WorkloadInfo:
    """Information about a GPU workload"""
    workload_id: str
    workload_type: str
    start_time: float
    estimated_duration: int
    actual_memory_usage: float = 0.0
    priority: int = 5  # 1-10, higher is more important
    allow_sharing: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def is_expired(self) -> bool:
        return self.elapsed_time > (self.estimated_duration * 2)
    
    @property
    def completion_percentage(self) -> float:
        if self.estimated_duration <= 0:
            return 0.0
        return min(100.0, (self.elapsed_time / self.estimated_duration) * 100)

class GPUStatus:
    """Enhanced GPU status tracking"""
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.is_available = False
        self.memory_free_gb = 0.0
        self.memory_total_gb = 0.0
        self.memory_used_gb = 0.0
        self.utilization_percent = 0.0
        self.temperature = 0.0
        self.power_draw = 0.0
        self.process_count = 0
        self.active_workloads: List[WorkloadInfo] = []
        self.last_updated = dt.now()
        self.error_count = 0
        self.total_allocations = 0
        self.successful_allocations = 0
        self.avg_workload_duration = 0.0
        self.peak_memory_usage = 0.0
        self.workload_history: deque = deque(maxlen=100)  # Keep last 100 workloads
        
    @property
    def memory_utilization_percent(self) -> float:
        if self.memory_total_gb == 0:
            return 100.0
        return (self.memory_used_gb / self.memory_total_gb) * 100
    
    @property
    def workload_count(self) -> int:
        return len(self.active_workloads)
    
    @property
    def allocation_success_rate(self) -> float:
        if self.total_allocations == 0:
            return 100.0
        return (self.successful_allocations / self.total_allocations) * 100
    
    @property
    def is_overloaded(self) -> bool:
        return (self.workload_count > 3 or 
                self.memory_utilization_percent > 90 or
                self.utilization_percent > 95)

class OptimizedDynamicGPUManager:
    """Complete GPU Manager with workload distribution and comprehensive monitoring"""
    
    def __init__(self, total_gpu_count: Optional[int] = None, 
                 memory_threshold: float = 0.7, 
                 utilization_threshold: float = 70.0,
                 config: Optional[GPUConfig] = None):
        """Initialize with auto-detection and safety for shared systems"""
        
        # Use provided config or create default
        if config is not None:
            self.config = config
        else:
            self.config = GPUConfig(
                total_gpu_count=total_gpu_count or 0,
                memory_threshold=memory_threshold,
                utilization_threshold=utilization_threshold
            )
        
        self.memory_threshold = self.config.memory_threshold
        self.utilization_threshold = self.config.utilization_threshold
        
        # Auto-detect GPU count if not provided
        if self.config.total_gpu_count == 0:
            self.config.total_gpu_count = self._detect_gpu_count()
        
        self.total_gpu_count = self.config.total_gpu_count
        self.available_gpu_count = self._get_available_gpu_count()
        
        # GPU status tracking
        self.gpu_status: Dict[int, GPUStatus] = {}
        self._status_lock = threading.Lock()
        self._last_refresh = 0
        self._refresh_interval = self.config.refresh_interval
        
        # Workload tracking
        self.active_workloads: Dict[int, List[WorkloadInfo]] = {}
        self.workload_history: List[Dict] = []
        self.workload_stats: Dict[str, Dict] = defaultdict(dict)
        
        # Performance tracking
        self.allocation_metrics = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'avg_allocation_time': 0.0,
            'peak_concurrent_workloads': 0
        }
        
        # Load balancing
        self.load_balancer_enabled = True
        self.auto_cleanup_enabled = True
        
        # Initialize GPU status
        self._initialize_gpu_status()
        
        # Start background cleanup if enabled
        if self.auto_cleanup_enabled:
            self._start_cleanup_thread()
        
        logger.info(f"✅ Complete GPU Manager: {self.available_gpu_count}/{self.total_gpu_count} GPUs available")
        logger.info(f"📊 Config: threshold={self.utilization_threshold}%, exclude_gpu_0={self.config.exclude_gpu_0}")
    
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
        start_gpu = 1 if (self.total_gpu_count > 1 and self.config.exclude_gpu_0) else 0
        
        for gpu_id in range(start_gpu, self.total_gpu_count):
            if self._is_gpu_usable(gpu_id):
                available += 1
        
        return available
    
    def _is_gpu_usable(self, gpu_id: int) -> bool:
        """Check if GPU is usable (quick check)"""
        try:
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                values = result.stdout.strip().split(',')
                if len(values) >= 3:
                    free_mb, util, temp = map(int, values)
                    free_gb = free_mb / 1024
                    
                    return (free_gb >= self.config.min_memory_gb and 
                           util < 90 and 
                           temp < self.config.temperature_threshold)
                
        except Exception as e:
            logger.warning(f"GPU {gpu_id} usability check failed: {e}")
        
        return False
    
    def _initialize_gpu_status(self):
        """Initialize status tracking for available GPUs"""
        start_gpu = 1 if (self.total_gpu_count > 1 and self.config.exclude_gpu_0) else 0
        
        for gpu_id in range(start_gpu, self.total_gpu_count):
            self.gpu_status[gpu_id] = GPUStatus(gpu_id)
            self.active_workloads[gpu_id] = []
        
        # Do initial status refresh
        self._refresh_gpu_status()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while self.auto_cleanup_enabled:
                try:
                    self.cleanup_completed_workloads()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Background cleanup thread started")
    
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
            self._update_system_metrics()
    
    def _update_gpu_status(self, gpu_id: int):
        """Update status for specific GPU"""
        try:
            # Get GPU info
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                values = result.stdout.strip().split(',')
                if len(values) >= 6:
                    total, used, free, util, temp, power = map(float, values)
                    
                    status = self.gpu_status[gpu_id]
                    status.memory_total_gb = total / 1024
                    status.memory_used_gb = used / 1024
                    status.memory_free_gb = free / 1024
                    status.utilization_percent = util
                    status.temperature = temp
                    status.power_draw = power
                    status.last_updated = dt.now()
                    
                    # Update peak memory usage
                    status.peak_memory_usage = max(status.peak_memory_usage, status.memory_used_gb)
                    
                    # Determine availability
                    status.is_available = (
                        status.memory_free_gb >= self.config.min_memory_gb and
                        status.utilization_percent < self.utilization_threshold and
                        status.temperature < self.config.temperature_threshold and
                        len(self.active_workloads[gpu_id]) < self.config.max_workloads_per_gpu
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
            self.gpu_status[gpu_id].is_available = False
    
    def _update_system_metrics(self):
        """Update system-wide metrics"""
        total_workloads = sum(len(workloads) for workloads in self.active_workloads.values())
        self.allocation_metrics['peak_concurrent_workloads'] = max(
            self.allocation_metrics['peak_concurrent_workloads'], 
            total_workloads
        )
    
    def get_optimal_gpu(self, workload_type: str = "default", 
                       min_memory_gb: float = None,
                       preferred_gpu: Optional[int] = None,
                       priority: int = 5) -> Optional[int]:
        """Get optimal GPU for workload with intelligent selection"""
        
        if min_memory_gb is None:
            min_memory_gb = self.config.min_memory_gb
        
        # Refresh status if needed
        self._refresh_gpu_status()
        
        candidates = []
        
        with self._status_lock:
            # Try preferred GPU first if specified
            if preferred_gpu is not None and preferred_gpu in self.gpu_status:
                status = self.gpu_status[preferred_gpu]
                if (status.is_available and 
                    status.memory_free_gb >= min_memory_gb and
                    not status.is_overloaded):
                    logger.info(f"Using preferred GPU {preferred_gpu} for {workload_type}")
                    return preferred_gpu
                else:
                    logger.warning(f"Preferred GPU {preferred_gpu} not available, finding alternative")
            
            # Find all suitable candidates
            for gpu_id, status in self.gpu_status.items():
                if not status.is_available:
                    continue
                
                if status.memory_free_gb < min_memory_gb:
                    continue
                
                if status.is_overloaded:
                    continue
                
                # Calculate score
                score = self._calculate_gpu_score(status, workload_type, priority)
                candidates.append((gpu_id, score, status))
        
        if not candidates:
            logger.warning(f"No suitable GPU found for {workload_type} (need {min_memory_gb:.1f}GB)")
            return None
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_gpu = candidates[0][0]
        best_score = candidates[0][1]
        
        logger.info(f"Selected GPU {best_gpu} (score: {best_score:.2f}) for {workload_type}")
        return best_gpu
    
    def _calculate_gpu_score(self, status: GPUStatus, workload_type: str, priority: int) -> float:
        """Calculate GPU suitability score with enhanced factors"""
        score = 0.0
        
        # Free memory weight (most important)
        score += status.memory_free_gb * 25
        
        # Lower utilization is better
        score += (100 - status.utilization_percent) * 0.8
        
        # Lower temperature is better
        score += (100 - status.temperature) * 0.3
        
        # Fewer active workloads is better
        workload_count = len(self.active_workloads[status.gpu_id])
        score += (self.config.max_workloads_per_gpu - workload_count) * 15
        
        # Fewer processes is better
        score += max(0, (20 - status.process_count) * 2)
        
        # Success rate bonus
        score += status.allocation_success_rate * 0.1
        
        # Power efficiency (lower power draw is better)
        if status.power_draw > 0:
            score += max(0, (400 - status.power_draw) * 0.05)
        
        # Workload type affinity (prefer GPUs that have handled this type well)
        if workload_type in self.workload_stats:
            gpu_stats = self.workload_stats[workload_type].get(str(status.gpu_id), {})
            success_rate = gpu_stats.get('success_rate', 50)  # Default to neutral
            score += (success_rate - 50) * 0.2  # Bonus/penalty based on historical performance
        
        # Priority adjustment
        score += priority * 2
        
        # Bonus for GPU 1,2,3 over GPU 0 in shared systems
        if status.gpu_id > 0 and self.config.exclude_gpu_0:
            score += 20
        
        return max(score, 0.0)
    
    def reserve_gpu_for_workload(self, workload_type: str, 
                                preferred_gpu: Optional[int] = None,
                                duration_estimate: int = 300,
                                allow_sharing: bool = True,
                                priority: int = 5,
                                metadata: Dict[str, Any] = None) -> Optional[int]:
        """Reserve GPU for workload with comprehensive tracking"""
        
        start_time = time.time()
        self.allocation_metrics['total_requests'] += 1
        
        try:
            # Try preferred GPU first
            if preferred_gpu is not None:
                if self._can_reserve_gpu(preferred_gpu, allow_sharing):
                    allocated_gpu = self._reserve_gpu(preferred_gpu, workload_type, duration_estimate, 
                                                    priority, allow_sharing, metadata)
                    if allocated_gpu is not None:
                        allocation_time = time.time() - start_time
                        self._update_allocation_metrics(True, allocation_time)
                        return allocated_gpu
            
            # Find optimal GPU
            optimal_gpu = self.get_optimal_gpu(workload_type, preferred_gpu=preferred_gpu, priority=priority)
            if optimal_gpu is not None:
                if self._can_reserve_gpu(optimal_gpu, allow_sharing):
                    allocated_gpu = self._reserve_gpu(optimal_gpu, workload_type, duration_estimate,
                                                    priority, allow_sharing, metadata)
                    if allocated_gpu is not None:
                        allocation_time = time.time() - start_time
                        self._update_allocation_metrics(True, allocation_time)
                        return allocated_gpu
            
            # Last resort: try any available GPU
            for gpu_id in self.gpu_status.keys():
                if self._can_reserve_gpu(gpu_id, True):  # Force allow sharing
                    logger.warning(f"Last resort allocation: GPU {gpu_id} for {workload_type}")
                    allocated_gpu = self._reserve_gpu(gpu_id, workload_type, duration_estimate,
                                                    priority, True, metadata)
                    if allocated_gpu is not None:
                        allocation_time = time.time() - start_time
                        self._update_allocation_metrics(True, allocation_time)
                        return allocated_gpu
            
            # Failed to allocate
            allocation_time = time.time() - start_time
            self._update_allocation_metrics(False, allocation_time)
            logger.warning(f"Could not reserve any GPU for {workload_type}")
            return None
            
        except Exception as e:
            allocation_time = time.time() - start_time
            self._update_allocation_metrics(False, allocation_time)
            logger.error(f"GPU reservation failed for {workload_type}: {e}")
            return None
    
    def _update_allocation_metrics(self, success: bool, allocation_time: float):
        """Update allocation metrics"""
        if success:
            self.allocation_metrics['successful_allocations'] += 1
        else:
            self.allocation_metrics['failed_allocations'] += 1
        
        # Update average allocation time
        total_allocations = self.allocation_metrics['total_requests']
        current_avg = self.allocation_metrics['avg_allocation_time']
        self.allocation_metrics['avg_allocation_time'] = (
            (current_avg * (total_allocations - 1) + allocation_time) / total_allocations
        )
    
    def _can_reserve_gpu(self, gpu_id: int, allow_sharing: bool) -> bool:
        """Check if GPU can be reserved"""
        if gpu_id not in self.gpu_status:
            return False
        
        status = self.gpu_status[gpu_id]
        
        if not status.is_available:
            return False
        
        if status.is_overloaded:
            return False
        
        workload_count = len(self.active_workloads[gpu_id])
        
        if not allow_sharing and workload_count > 0:
            return False
        
        if workload_count >= self.config.max_workloads_per_gpu:
            return False
        
        return True
    
    def _reserve_gpu(self, gpu_id: int, workload_type: str, duration: int,
                    priority: int, allow_sharing: bool, metadata: Dict[str, Any]) -> Optional[int]:
        """Actually reserve the GPU"""
        try:
            workload_id = f"{workload_type}_{gpu_id}_{int(time.time())}"
            
            workload_info = WorkloadInfo(
                workload_id=workload_id,
                workload_type=workload_type,
                start_time=time.time(),
                estimated_duration=duration,
                priority=priority,
                allow_sharing=allow_sharing,
                metadata=metadata or {}
            )
            
            self.active_workloads[gpu_id].append(workload_info)
            self.gpu_status[gpu_id].total_allocations += 1
            self.gpu_status[gpu_id].successful_allocations += 1
            
            logger.info(f"Reserved GPU {gpu_id} for {workload_type} (ID: {workload_id}, priority: {priority})")
            return gpu_id
            
        except Exception as e:
            logger.error(f"Failed to reserve GPU {gpu_id}: {e}")
            return None
    
    def release_gpu_workload(self, gpu_id: int, workload_identifier: str):
        """Release GPU workload with comprehensive cleanup"""
        if gpu_id not in self.active_workloads:
            logger.warning(f"GPU {gpu_id} not found in active workloads")
            return
        
        original_count = len(self.active_workloads[gpu_id])
        released_workloads = []
        
        # Find and remove matching workloads
        remaining_workloads = []
        for workload in self.active_workloads[gpu_id]:
            if (workload_identifier in workload.workload_id or 
                workload_identifier in workload.workload_type):
                released_workloads.append(workload)
                # Add to history
                self.gpu_status[gpu_id].workload_history.append({
                    'workload_id': workload.workload_id,
                    'workload_type': workload.workload_type,
                    'duration': workload.elapsed_time,
                    'completed_normally': True,
                    'end_time': time.time()
                })
            else:
                remaining_workloads.append(workload)
        
        self.active_workloads[gpu_id] = remaining_workloads
        released_count = len(released_workloads)
        
        if released_count > 0:
            # Update workload statistics
            for workload in released_workloads:
                self._update_workload_stats(workload, gpu_id, success=True)
            
            logger.info(f"Released {released_count} workload(s) from GPU {gpu_id}")
        else:
            logger.warning(f"No matching workloads found for identifier '{workload_identifier}' on GPU {gpu_id}")
    
    def _update_workload_stats(self, workload: WorkloadInfo, gpu_id: int, success: bool):
        """Update workload statistics for performance tracking"""
        workload_type = workload.workload_type
        
        if workload_type not in self.workload_stats:
            self.workload_stats[workload_type] = {}
        
        gpu_key = str(gpu_id)
        if gpu_key not in self.workload_stats[workload_type]:
            self.workload_stats[workload_type][gpu_key] = {
                'total_runs': 0,
                'successful_runs': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.workload_stats[workload_type][gpu_key]
        stats['total_runs'] += 1
        if success:
            stats['successful_runs'] += 1
        
        duration = workload.elapsed_time
        stats['total_duration'] += duration
        stats['avg_duration'] = stats['total_duration'] / stats['total_runs']
        stats['success_rate'] = (stats['successful_runs'] / stats['total_runs']) * 100
    
    def cleanup_completed_workloads(self):
        """Clean up workloads that should have completed"""
        current_time = time.time()
        total_cleaned = 0
        
        for gpu_id in self.active_workloads:
            original_count = len(self.active_workloads[gpu_id])
            expired_workloads = []
            remaining_workloads = []
            
            for workload in self.active_workloads[gpu_id]:
                if workload.is_expired:
                    expired_workloads.append(workload)
                    # Add to history as expired
                    self.gpu_status[gpu_id].workload_history.append({
                        'workload_id': workload.workload_id,
                        'workload_type': workload.workload_type,
                        'duration': workload.elapsed_time,
                        'completed_normally': False,
                        'expired': True,
                        'end_time': current_time
                    })
                else:
                    remaining_workloads.append(workload)
            
            self.active_workloads[gpu_id] = remaining_workloads
            cleaned_count = len(expired_workloads)
            total_cleaned += cleaned_count
            
            if cleaned_count > 0:
                # Update stats for expired workloads
                for workload in expired_workloads:
                    self._update_workload_stats(workload, gpu_id, success=False)
                
                logger.info(f"Cleaned {cleaned_count} expired workloads from GPU {gpu_id}")
        
        if total_cleaned > 0:
            logger.info(f"🧹 Total cleanup: {total_cleaned} expired workloads across all GPUs")
    
    def get_workload_distribution(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get current workload distribution across GPUs"""
        distribution = {}
        
        with self._status_lock:
            for gpu_id, workloads in self.active_workloads.items():
                distribution[f"gpu_{gpu_id}"] = []
                for workload in workloads:
                    distribution[f"gpu_{gpu_id}"].append({
                        'workload_id': workload.workload_id,
                        'workload_type': workload.workload_type,
                        'elapsed_time': workload.elapsed_time,
                        'estimated_duration': workload.estimated_duration,
                        'completion_percentage': workload.completion_percentage,
                        'priority': workload.priority,
                        'allow_sharing': workload.allow_sharing,
                        'memory_usage': workload.actual_memory_usage,
                        'metadata': workload.metadata
                    })
        
        return distribution
    
    def get_gpu_status_detailed(self) -> Dict[str, Any]:
        """Get detailed status of all GPUs"""
        self._refresh_gpu_status()
        
        status_dict = {}
        
        with self._status_lock:
            for gpu_id, status in self.gpu_status.items():
                status_dict[f"gpu_{gpu_id}"] = {
                    "gpu_id": gpu_id,
                    "is_available": status.is_available,
                    "memory_free_gb": round(status.memory_free_gb, 2),
                    "memory_total_gb": round(status.memory_total_gb, 2),
                    "memory_used_gb": round(status.memory_used_gb, 2),
                    "memory_utilization_percent": round(status.memory_utilization_percent, 1),
                    "utilization_percent": round(status.utilization_percent, 1),
                    "temperature": status.temperature,
                    "power_draw": status.power_draw,
                    "process_count": status.process_count,
                    "active_workloads": status.workload_count,
                    "workload_details": [asdict(w) for w in status.active_workloads],
                    "last_updated": status.last_updated.isoformat(),
                    "error_count": status.error_count,
                    "total_allocations": status.total_allocations,
                    "allocation_success_rate": round(status.allocation_success_rate, 1),
                    "avg_workload_duration": round(status.avg_workload_duration, 2),
                    "peak_memory_usage": round(status.peak_memory_usage, 2),
                    "is_overloaded": status.is_overloaded,
                    "recent_workloads": list(status.workload_history)[-5:]  # Last 5 workloads
                }
        
        return status_dict
    
    def get_system_gpu_summary(self) -> Dict[str, Any]:
        """Get high-level GPU system summary with comprehensive metrics"""
        self._refresh_gpu_status()
        
        available_gpus = sum(1 for status in self.gpu_status.values() if status.is_available)
        total_workloads = sum(len(workloads) for workloads in self.active_workloads.values())
        overloaded_gpus = sum(1 for status in self.gpu_status.values() if status.is_overloaded)
        
        # Calculate system-wide averages
        if self.gpu_status:
            avg_utilization = sum(s.utilization_percent for s in self.gpu_status.values()) / len(self.gpu_status)
            avg_temperature = sum(s.temperature for s in self.gpu_status.values()) / len(self.gpu_status)
            total_memory_free = sum(s.memory_free_gb for s in self.gpu_status.values())
            total_memory_total = sum(s.memory_total_gb for s in self.gpu_status.values())
            avg_memory_utilization = ((total_memory_total - total_memory_free) / total_memory_total) * 100 if total_memory_total > 0 else 0
        else:
            avg_utilization = avg_temperature = total_memory_free = avg_memory_utilization = 0
        
        # System health assessment
        if available_gpus == 0:
            health_status = "critical"
        elif overloaded_gpus > len(self.gpu_status) / 2:
            health_status = "degraded"
        elif available_gpus < len(self.gpu_status) / 2:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "total_gpus": self.total_gpu_count,
            "managed_gpus": len(self.gpu_status),
            "available_gpus": available_gpus,
            "overloaded_gpus": overloaded_gpus,
            "total_active_workloads": total_workloads,
            "system_health": health_status,
            "averages": {
                "gpu_utilization_percent": round(avg_utilization, 1),
                "temperature": round(avg_temperature, 1),
                "memory_utilization_percent": round(avg_memory_utilization, 1)
            },
            "totals": {
                "memory_free_gb": round(total_memory_free, 2),
                "memory_total_gb": round(total_memory_total, 2)
            },
            "allocation_metrics": self.allocation_metrics.copy(),
            "config": asdict(self.config),
            "last_updated": dt.now().isoformat()
        }
    
    def get_workload_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workload statistics"""
        stats = {
            "by_type": {},
            "by_gpu": {},
            "system_totals": {
                "active_workloads": sum(len(w) for w in self.active_workloads.values()),
                "total_workload_types": len(self.workload_stats),
                "avg_workload_duration": 0.0,
                "success_rate": 0.0
            }
        }
        
        # Aggregate statistics by workload type
        total_runs = 0
        total_successful = 0
        total_duration = 0.0
        
        for workload_type, gpu_stats in self.workload_stats.items():
            type_stats = {
                "total_runs": 0,
                "successful_runs": 0,
                "avg_duration": 0.0,
                "success_rate": 0.0,
                "gpu_distribution": {}
            }
            
            for gpu_id, gpu_data in gpu_stats.items():
                type_stats["total_runs"] += gpu_data["total_runs"]
                type_stats["successful_runs"] += gpu_data["successful_runs"]
                total_duration += gpu_data["total_duration"]
                type_stats["gpu_distribution"][gpu_id] = gpu_data.copy()
                
                total_runs += gpu_data["total_runs"]
                total_successful += gpu_data["successful_runs"]
            
            if type_stats["total_runs"] > 0:
                type_stats["avg_duration"] = total_duration / type_stats["total_runs"]
                type_stats["success_rate"] = (type_stats["successful_runs"] / type_stats["total_runs"]) * 100
            
            stats["by_type"][workload_type] = type_stats
        
        # Aggregate statistics by GPU
        for gpu_id, status in self.gpu_status.items():
            gpu_stats = {
                "active_workloads": len(self.active_workloads[gpu_id]),
                "total_allocations": status.total_allocations,
                "allocation_success_rate": status.allocation_success_rate,
                "workload_types_handled": set(),
                "avg_workload_duration": 0.0,
                "current_utilization": status.utilization_percent,
                "memory_utilization": status.memory_utilization_percent
            }
            
            # Calculate workload types handled by this GPU
            total_gpu_duration = 0.0
            total_gpu_runs = 0
            
            for workload_type, type_stats in self.workload_stats.items():
                if str(gpu_id) in type_stats:
                    gpu_stats["workload_types_handled"].add(workload_type)
                    gpu_data = type_stats[str(gpu_id)]
                    total_gpu_duration += gpu_data["total_duration"]
                    total_gpu_runs += gpu_data["total_runs"]
            
            if total_gpu_runs > 0:
                gpu_stats["avg_workload_duration"] = total_gpu_duration / total_gpu_runs
            
            gpu_stats["workload_types_handled"] = list(gpu_stats["workload_types_handled"])
            stats["by_gpu"][f"gpu_{gpu_id}"] = gpu_stats
        
        # System totals
        if total_runs > 0:
            stats["system_totals"]["avg_workload_duration"] = total_duration / total_runs
            stats["system_totals"]["success_rate"] = (total_successful / total_runs) * 100
        
        return stats
    
    def force_refresh(self):
        """Force immediate status refresh"""
        self._refresh_gpu_status(force=True)
    
    def get_recommendation(self, workload_type: str, min_memory_gb: float = None) -> Dict[str, Any]:
        """Get GPU recommendation for workload type with detailed reasoning"""
        if min_memory_gb is None:
            min_memory_gb = self.config.min_memory_gb
        
        optimal_gpu = self.get_optimal_gpu(workload_type, min_memory_gb=min_memory_gb)
        
        if optimal_gpu is None:
            # Analyze why no GPU is available
            reasons = []
            alternatives = []
            
            for gpu_id, status in self.gpu_status.items():
                if not status.is_available:
                    if status.memory_free_gb < min_memory_gb:
                        reasons.append(f"GPU {gpu_id}: Insufficient memory ({status.memory_free_gb:.1f}GB < {min_memory_gb:.1f}GB)")
                    if status.utilization_percent >= self.utilization_threshold:
                        reasons.append(f"GPU {gpu_id}: High utilization ({status.utilization_percent:.1f}%)")
                    if status.temperature >= self.config.temperature_threshold:
                        reasons.append(f"GPU {gpu_id}: High temperature ({status.temperature:.1f}°C)")
                    if status.is_overloaded:
                        reasons.append(f"GPU {gpu_id}: Overloaded ({status.workload_count} workloads)")
                else:
                    alternatives.append(gpu_id)
            
            # Estimate wait time based on current workloads
            min_wait_time = float('inf')
            for gpu_id, workloads in self.active_workloads.items():
                if workloads:
                    earliest_completion = min(
                        (w.estimated_duration - w.elapsed_time) for w in workloads
                        if (w.estimated_duration - w.elapsed_time) > 0
                    )
                    min_wait_time = min(min_wait_time, earliest_completion)
            
            wait_time = int(min_wait_time) if min_wait_time != float('inf') else None
            
            return {
                "recommended_gpu": None,
                "reason": "No suitable GPU available",
                "detailed_reasons": reasons,
                "alternatives": alternatives,
                "wait_suggested": True,
                "estimated_wait_time_seconds": wait_time,
                "suggestion": "Consider reducing memory requirements or waiting for current workloads to complete"
            }
        
        status = self.gpu_status[optimal_gpu]
        
        # Get historical performance for this workload type on this GPU
        historical_performance = ""
        if workload_type in self.workload_stats:
            gpu_stats = self.workload_stats[workload_type].get(str(optimal_gpu), {})
            if gpu_stats.get('total_runs', 0) > 0:
                historical_performance = (f"Historical: {gpu_stats['success_rate']:.1f}% success rate, "
                                        f"{gpu_stats['avg_duration']:.1f}s avg duration")
        
        return {
            "recommended_gpu": optimal_gpu,
            "reason": f"Best available: {status.memory_free_gb:.1f}GB free, {status.utilization_percent:.1f}% util",
            "memory_available": status.memory_free_gb,
            "current_utilization": status.utilization_percent,
            "temperature": status.temperature,
            "active_workloads": len(self.active_workloads[optimal_gpu]),
            "estimated_wait_time": 0,
            "allocation_success_rate": status.allocation_success_rate,
            "historical_performance": historical_performance,
            "confidence": "high" if status.allocation_success_rate > 90 else "medium" if status.allocation_success_rate > 70 else "low"
        }
    
    def rebalance_workloads(self) -> Dict[str, Any]:
        """Attempt to rebalance workloads across GPUs"""
        if not self.load_balancer_enabled:
            return {"status": "disabled", "message": "Load balancing is disabled"}
        
        # Find overloaded and underutilized GPUs
        overloaded_gpus = []
        underutilized_gpus = []
        
        for gpu_id, status in self.gpu_status.items():
            if status.is_overloaded:
                overloaded_gpus.append((gpu_id, status))
            elif status.is_available and status.workload_count < 2:
                underutilized_gpus.append((gpu_id, status))
        
        if not overloaded_gpus or not underutilized_gpus:
            return {
                "status": "no_action_needed",
                "overloaded_gpus": len(overloaded_gpus),
                "underutilized_gpus": len(underutilized_gpus)
            }
        
        # This is a placeholder for actual workload migration
        # In practice, you'd need to implement workload migration logic
        # which is complex and depends on the specific workload types
        
        suggestions = []
        for gpu_id, status in overloaded_gpus:
            for target_gpu_id, target_status in underutilized_gpus:
                if target_status.memory_free_gb > 4.0:  # Has enough memory
                    suggestions.append({
                        "from_gpu": gpu_id,
                        "to_gpu": target_gpu_id,
                        "reason": f"Move workload from overloaded GPU {gpu_id} to underutilized GPU {target_gpu_id}"
                    })
                    break
        
        return {
            "status": "suggestions_available",
            "suggestions": suggestions,
            "overloaded_gpus": len(overloaded_gpus),
            "underutilized_gpus": len(underutilized_gpus),
            "note": "Automatic workload migration not implemented - manual intervention required"
        }
    
    def get_available_gpu(self, preferred_gpu: Optional[int] = None, 
                         fallback: bool = True, allow_sharing: bool = True) -> Optional[int]:
        """Legacy method for backward compatibility"""
        return self.get_optimal_gpu("default", preferred_gpu=preferred_gpu)
    
    def enable_load_balancing(self, enabled: bool = True):
        """Enable or disable load balancing"""
        self.load_balancer_enabled = enabled
        logger.info(f"Load balancing {'enabled' if enabled else 'disabled'}")
    
    def enable_auto_cleanup(self, enabled: bool = True):
        """Enable or disable automatic cleanup"""
        self.auto_cleanup_enabled = enabled
        logger.info(f"Auto cleanup {'enabled' if enabled else 'disabled'}")
    
    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export comprehensive metrics in specified format"""
        metrics = {
            "timestamp": dt.now().isoformat(),
            "system_summary": self.get_system_gpu_summary(),
            "gpu_status": self.get_gpu_status_detailed(),
            "workload_distribution": self.get_workload_distribution(),
            "workload_statistics": self.get_workload_statistics(),
            "config": asdict(self.config)
        }
        
        if format_type.lower() == "json":
            return json.dumps(metrics, indent=2, default=str)
        else:
            return metrics
    
    def reset_statistics(self):
        """Reset all statistics and metrics"""
        self.allocation_metrics = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'avg_allocation_time': 0.0,
            'peak_concurrent_workloads': 0
        }
        
        self.workload_stats.clear()
        
        for status in self.gpu_status.values():
            status.total_allocations = 0
            status.successful_allocations = 0
            status.avg_workload_duration = 0.0
            status.peak_memory_usage = 0.0
            status.workload_history.clear()
            status.error_count = 0
        
        logger.info("🔄 All GPU manager statistics reset")
    
    def shutdown(self):
        """Clean shutdown with comprehensive cleanup"""
        logger.info("🔄 Shutting down GPU manager...")
        
        # Disable auto cleanup
        self.auto_cleanup_enabled = False
        
        # Clear all workloads
        total_workloads = 0
        for gpu_id in self.active_workloads:
            total_workloads += len(self.active_workloads[gpu_id])
            self.active_workloads[gpu_id].clear()
        
        if total_workloads > 0:
            logger.warning(f"Forcibly cleared {total_workloads} active workloads during shutdown")
        
        # Clear status
        self.gpu_status.clear()
        
        # Clear statistics
        self.workload_stats.clear()
        self.workload_history.clear()
        
        logger.info("✅ GPU manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
    
    def __repr__(self):
        return (f"OptimizedDynamicGPUManager("
                f"gpus={self.total_gpu_count}, "
                f"available={self.available_gpu_count}, "
                f"active_workloads={sum(len(w) for w in self.active_workloads.values())}, "
                f"health={self.get_system_gpu_summary()['system_health']})")


# For backward compatibility
DynamicGPUManager = OptimizedDynamicGPUManager


# Utility functions for easy access
def create_gpu_manager(config: Optional[GPUConfig] = None) -> OptimizedDynamicGPUManager:
    """Create a GPU manager with optional configuration"""
    return OptimizedDynamicGPUManager(config=config)


def get_default_gpu_config() -> GPUConfig:
    """Get default GPU configuration"""
    return GPUConfig()


def create_shared_server_config() -> GPUConfig:
    """Create GPU configuration optimized for shared servers"""
    return GPUConfig(
        exclude_gpu_0=True,
        memory_threshold=0.6,  # More conservative
        utilization_threshold=60.0,  # Lower threshold
        min_memory_gb=1.5,  # Lower requirement
        max_workloads_per_gpu=2,  # Fewer concurrent workloads
        temperature_threshold=80.0,  # Lower temperature threshold
        refresh_interval=20  # More frequent updates
    )
DynamicGPUManager = OptimizedDynamicGPUManager
