# utils/gpu_manager.py
"""
Enhanced GPU Manager for dynamic GPU allocation and workload distribution
"""

import torch
import psutil
import logging
from typing import Dict, List, Optional, Any, Union
import time
import threading
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import queue

class GPUStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OCCUPIED = "occupied"
    ERROR = "error"

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

class DynamicGPUManager:
    """Enhanced GPU Manager with dynamic allocation"""
    
    def __init__(self, total_gpu_count: int = 4, memory_threshold: float = 0.85, utilization_threshold: float = 80.0):
        self.total_gpu_count = total_gpu_count
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self.logger = logging.getLogger(__name__)
        
        # GPU tracking
        self.gpu_info = {}
        self.workload_assignments = defaultdict(list)
        self.gpu_reservation_queue = queue.PriorityQueue()
        self.lock = threading.RLock()
        
        # Monitoring
        self.monitoring_active = True
        self.monitoring_interval = 5  # seconds
        
        self._initialize_gpu_monitoring()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring system"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. GPU management disabled.")
            return
        
        available_gpus = torch.cuda.device_count()
        self.total_gpu_count = min(self.total_gpu_count, available_gpus)
        
        self.logger.info(f"Initialized dynamic GPU manager with {self.total_gpu_count} GPUs")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_gpus_continuously, 
            daemon=True,
            name="GPU_Monitor"
        )
        self.monitoring_thread.start()
    
    def _monitor_gpus_continuously(self):
        """Continuously monitor all GPU states"""
        while self.monitoring_active:
            try:
                self._update_all_gpu_info()
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {str(e)}")
                time.sleep(self.monitoring_interval * 2)  # Wait longer on error
    
    def _update_all_gpu_info(self):
        """Update information for all GPUs"""
        if not torch.cuda.is_available():
            return
            
        with self.lock:
            for gpu_id in range(self.total_gpu_count):
                try:
                    gpu_info = self._get_gpu_info(gpu_id)
                    self.gpu_info[gpu_id] = gpu_info
                    
                except Exception as e:
                    self.logger.error(f"Error updating GPU {gpu_id}: {str(e)}")
                    self.gpu_info[gpu_id] = GPUInfo(
                        gpu_id=gpu_id,
                        name="Unknown",
                        total_memory=0,
                        used_memory=0,
                        free_memory=0,
                        utilization=100.0,
                        temperature=0,
                        status=GPUStatus.ERROR,
                        process_count=0,
                        last_updated=time.time()
                    )
    
    def _get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get detailed information for a specific GPU"""
        try:
            with torch.cuda.device(gpu_id):
                # Memory information
                memory_info = torch.cuda.mem_get_info()
                free_memory = memory_info[0]
                total_memory = memory_info[1]
                used_memory = total_memory - free_memory
                
                # Calculate utilization
                memory_utilization = (used_memory / total_memory) * 100
                
                # Get device properties
                props = torch.cuda.get_device_properties(gpu_id)
                
                # Determine GPU status
                if memory_utilization > self.utilization_threshold:
                    status = GPUStatus.OCCUPIED
                elif memory_utilization > 50:
                    status = GPUStatus.BUSY
                else:
                    status = GPUStatus.AVAILABLE
                
                # Count processes (simplified)
                process_count = 1 if used_memory > total_memory * 0.1 else 0
                
                return GPUInfo(
                    gpu_id=gpu_id,
                    name=props.name,
                    total_memory=total_memory,
                    used_memory=used_memory,
                    free_memory=free_memory,
                    utilization=memory_utilization,
                    temperature=0,  # Would need nvidia-ml-py for real temperature
                    status=status,
                    process_count=process_count,
                    last_updated=time.time()
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get GPU {gpu_id} info: {str(e)}")
            raise
    
    def get_available_gpu(self, preferred_gpu: Optional[int] = None, fallback: bool = True) -> Optional[int]:
        """
        Get the best available GPU with preference system
        
        Args:
            preferred_gpu: Preferred GPU ID (0, 1, 2, 3)
            fallback: Whether to fallback to other GPUs if preferred is unavailable
            
        Returns:
            GPU ID or None if no GPU available
        """
        with self.lock:
            # Check if preferred GPU is available
            if preferred_gpu is not None and preferred_gpu in self.gpu_info:
                gpu_info = self.gpu_info[preferred_gpu]
                if gpu_info.status == GPUStatus.AVAILABLE:
                    self.logger.info(f"Using preferred GPU {preferred_gpu}")
                    return preferred_gpu
                elif not fallback:
                    self.logger.warning(f"Preferred GPU {preferred_gpu} not available and fallback disabled")
                    return None
            
            # Find best available GPU
            available_gpus = [
                (gpu_id, info) for gpu_id, info in self.gpu_info.items()
                if info.status == GPUStatus.AVAILABLE
            ]
            
            if available_gpus:
                # Sort by free memory (descending)
                best_gpu = max(available_gpus, key=lambda x: x[1].free_memory)
                self.logger.info(f"Selected GPU {best_gpu[0]} (free memory: {best_gpu[1].free_memory / (1024**3):.1f}GB)")
                return best_gpu[0]
            
            # If no fully available GPU, try busy but usable GPUs
            busy_gpus = [
                (gpu_id, info) for gpu_id, info in self.gpu_info.items()
                if info.status == GPUStatus.BUSY and info.utilization < self.utilization_threshold
            ]
            
            if busy_gpus:
                best_gpu = min(busy_gpus, key=lambda x: x[1].utilization)
                self.logger.warning(f"Using busy GPU {best_gpu[0]} (utilization: {best_gpu[1].utilization:.1f}%)")
                return best_gpu[0]
            
            # Last resort: use least utilized GPU
            if self.gpu_info:
                last_resort = min(self.gpu_info.items(), key=lambda x: x[1].utilization)
                self.logger.warning(f"Using last resort GPU {last_resort[0]} (utilization: {last_resort[1].utilization:.1f}%)")
                return last_resort[0]
            
            self.logger.error("No GPUs available")
            return None
    
    def reserve_gpu_for_workload(self, workload_type: str, preferred_gpu: Optional[int] = None, 
                                duration_estimate: int = 300) -> Optional[int]:
        """
        Reserve a GPU for a specific workload
        
        Args:
            workload_type: Type of workload (e.g., 'llm_inference', 'embedding', 'analysis')
            preferred_gpu: Preferred GPU ID
            duration_estimate: Estimated duration in seconds
            
        Returns:
            Reserved GPU ID or None
        """
        gpu_id = self.get_available_gpu(preferred_gpu)
        
        if gpu_id is not None:
            with self.lock:
                self.workload_assignments[gpu_id].append({
                    "type": workload_type,
                    "start_time": time.time(),
                    "estimated_duration": duration_estimate,
                    "status": "active"
                })
            
            self.logger.info(f"Reserved GPU {gpu_id} for {workload_type} (estimated {duration_estimate}s)")
            
        return gpu_id
    
    def release_gpu_workload(self, gpu_id: int, workload_type: str):
        """Release a GPU workload"""
        with self.lock:
            if gpu_id in self.workload_assignments:
                workloads = self.workload_assignments[gpu_id]
                for workload in workloads:
                    if workload["type"] == workload_type and workload["status"] == "active":
                        workload["status"] = "completed"
                        workload["end_time"] = time.time()
                        self.logger.info(f"Released GPU {gpu_id} from {workload_type}")
                        break
    
    def get_gpu_status_detailed(self) -> Dict[str, Any]:
        """Get detailed status of all GPUs"""
        with self.lock:
            return {
                f"gpu_{gpu_id}": {
                    "name": info.name,
                    "status": info.status.value,
                    "utilization_percent": round(info.utilization, 2),
                    "memory_used_gb": round(info.used_memory / (1024**3), 2),
                    "memory_total_gb": round(info.total_memory / (1024**3), 2),
                    "memory_free_gb": round(info.free_memory / (1024**3), 2),
                    "memory_utilization_percent": round((info.used_memory / info.total_memory) * 100, 2),
                    "process_count": info.process_count,
                    "is_available": info.status == GPUStatus.AVAILABLE,
                    "active_workloads": len([w for w in self.workload_assignments.get(gpu_id, []) if w["status"] == "active"]),
                    "last_updated": info.last_updated
                }
                for gpu_id, info in self.gpu_info.items()
            }
    
    def get_workload_distribution(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get current workload distribution across GPUs"""
        with self.lock:
            return {
                gpu_id: [
                    {
                        "type": workload["type"],
                        "duration": time.time() - workload["start_time"] if workload["status"] == "active" else workload.get("end_time", 0) - workload["start_time"],
                        "status": workload["status"]
                    }
                    for workload in workloads
                ]
                for gpu_id, workloads in self.workload_assignments.items()
            }
    
    def cleanup_completed_workloads(self):
        """Clean up completed workloads from tracking"""
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour
        
        with self.lock:
            for gpu_id in self.workload_assignments:
                self.workload_assignments[gpu_id] = [
                    workload for workload in self.workload_assignments[gpu_id]
                    if workload["status"] == "active" or 
                    (current_time - workload.get("end_time", 0)) < cleanup_threshold
                ]
    
    def get_recommendation(self, workload_type: str) -> Dict[str, Any]:
        """Get GPU recommendation for a specific workload type"""
        with self.lock:
            recommendations = []
            
            for gpu_id, info in self.gpu_info.items():
                score = 0
                
                # Base score on availability
                if info.status == GPUStatus.AVAILABLE:
                    score += 100
                elif info.status == GPUStatus.BUSY:
                    score += 50
                elif info.status == GPUStatus.OCCUPIED:
                    score += 10
                else:
                    score = 0
                
                # Adjust for memory availability
                memory_factor = (info.free_memory / info.total_memory) * 50
                score += memory_factor
                
                # Adjust for current workload
                active_workloads = len([w for w in self.workload_assignments.get(gpu_id, []) if w["status"] == "active"])
                score -= active_workloads * 10
                
                recommendations.append({
                    "gpu_id": gpu_id,
                    "score": score,
                    "status": info.status.value,
                    "free_memory_gb": info.free_memory / (1024**3),
                    "utilization": info.utilization,
                    "active_workloads": active_workloads
                })
            
            # Sort by score (highest first)
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "workload_type": workload_type,
                "recommended_gpu": recommendations[0]["gpu_id"] if recommendations else None,
                "all_options": recommendations
            }
    
    def force_refresh(self):
        """Force immediate refresh of GPU information"""
        self._update_all_gpu_info()
    
    def shutdown(self):
        """Shutdown the GPU manager"""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5)
        self.logger.info("GPU Manager shutdown complete")


# Context manager for GPU allocation
class GPUContext:
    """Context manager for automatic GPU allocation and release"""
    
    def __init__(self, gpu_manager: DynamicGPUManager, workload_type: str, 
                 preferred_gpu: Optional[int] = None, duration_estimate: int = 300):
        self.gpu_manager = gpu_manager
        self.workload_type = workload_type
        self.preferred_gpu = preferred_gpu
        self.duration_estimate = duration_estimate
        self.allocated_gpu = None
    
    def __enter__(self) -> Optional[int]:
        self.allocated_gpu = self.gpu_manager.reserve_gpu_for_workload(
            self.workload_type, self.preferred_gpu, self.duration_estimate
        )
        return self.allocated_gpu
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.allocated_gpu is not None:
            self.gpu_manager.release_gpu_workload(self.allocated_gpu, self.workload_type)