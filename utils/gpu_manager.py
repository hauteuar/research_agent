# utils/gpu_manager.py
"""
GPU Manager for distributing workloads across multiple GPUs
"""

import torch
import psutil
import logging
from typing import Dict, List, Optional, Any
import time
import threading
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GPUStatus:
    gpu_id: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int
    utilization: float
    temperature: float
    is_available: bool

class GPUManager:
    """Manages GPU resources and workload distribution"""
    
    def __init__(self, gpu_count: int = 3):
        self.gpu_count = gpu_count
        self.logger = logging.getLogger(__name__)
        self.gpu_stats = {}
        self.workload_queue = defaultdict(list)
        self.lock = threading.Lock()
        
        self._initialize_gpu_monitoring()
    
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. GPU management disabled.")
            return
        
        available_gpus = torch.cuda.device_count()
        self.gpu_count = min(self.gpu_count, available_gpus)
        
        self.logger.info(f"Initialized GPU manager with {self.gpu_count} GPUs")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_gpus, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_gpus(self):
        """Monitor GPU status continuously"""
        while True:
            try:
                if torch.cuda.is_available():
                    for gpu_id in range(self.gpu_count):
                        with torch.cuda.device(gpu_id):
                            # Get memory info
                            memory_info = torch.cuda.mem_get_info()
                            free_memory = memory_info[0]
                            total_memory = memory_info[1]
                            used_memory = total_memory - free_memory
                            
                            # Get utilization (simplified)
                            utilization = (used_memory / total_memory) * 100
                            
                            # Get GPU properties
                            props = torch.cuda.get_device_properties(gpu_id)
                            
                            self.gpu_stats[gpu_id] = GPUStatus(
                                gpu_id=gpu_id,
                                name=props.name,
                                total_memory=total_memory,
                                used_memory=used_memory,
                                free_memory=free_memory,
                                utilization=utilization,
                                temperature=0,  # Would need nvidia-ml-py for real temperature
                                is_available=utilization < 90  # Consider available if < 90% used
                            )
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {str(e)}")
                time.sleep(30)  # Wait longer on error
    
    def get_available_gpu(self) -> Optional[int]:
        """Get the most available GPU"""
        with self.lock:
            if not self.gpu_stats:
                return 0  # Default to GPU 0 if no stats available
            
            available_gpus = [
                (gpu_id, status) for gpu_id, status in self.gpu_stats.items()
                if status.is_available
            ]
            
            if not available_gpus:
                # Return least utilized GPU
                return min(self.gpu_stats.keys(), 
                          key=lambda x: self.gpu_stats[x].utilization)
            
            # Return GPU with most free memory
            return min(available_gpus, key=lambda x: x[1].used_memory)[0]
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get status of all GPUs"""
        return {
            f"gpu_{gpu_id}": {
                "name": status.name,
                "utilization": status.utilization,
                "memory_used_gb": status.used_memory / (1024**3),
                "memory_total_gb": status.total_memory / (1024**3),
                "memory_free_gb": status.free_memory / (1024**3),
                "is_available": status.is_available
            }
            for gpu_id, status in self.gpu_stats.items()
        }
    
    def assign_workload(self, workload_type: str, gpu_preference: int = None) -> int:
        """Assign workload to optimal GPU"""
        if gpu_preference is not None and gpu_preference < self.gpu_count:
            if self.gpu_stats.get(gpu_preference, {}).is_available:
                return gpu_preference
        
        # Find optimal GPU
        optimal_gpu = self.get_available_gpu()
        
        # Track workload assignment
        with self.lock:
            self.workload_queue[optimal_gpu].append({
                "type": workload_type,
                "timestamp": time.time()
            })
        
        return optimal_gpu


# utils/health_monitor.py
"""
System Health Monitor
"""

import psutil
import logging
import time
import threading
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import deque

class HealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self, history_size: int = 100):
        self.logger = logging.getLogger(__name__)
        self.history_size = history_size
        
        # Health metrics history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        
        # System alerts
        self.alerts = []
        self.alert_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "response_time": 30  # seconds
        }
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Continuously monitor system metrics"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_history.append((timestamp, cpu_percent))
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.memory_history.append((timestamp, memory.percent))
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.disk_history.append((timestamp, disk_percent))
                
                # Check thresholds and generate alerts
                self._check_thresholds(cpu_percent, memory.percent, disk_percent)
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                time.sleep(60)
    
    def _check_thresholds(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check if metrics exceed thresholds"""
        alerts = []
        
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu_high",
                "message": f"CPU usage high: {cpu_percent:.1f}%",
                "severity": "warning" if cpu_percent < 95 else "critical",
                "timestamp": datetime.now()
            })
        
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "memory_high",
                "message": f"Memory usage high: {memory_percent:.1f}%",
                "severity": "warning" if memory_percent < 95 else "critical",
                "timestamp": datetime.now()
            })
        
        if disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "disk_high",
                "message": f"Disk usage high: {disk_percent:.1f}%",
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts 
            if alert["timestamp"] > cutoff_time
        ]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Current metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process count
            process_count = len(psutil.pids())
            
            # Network I/O
            network = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "status": "normal" if cpu_percent < 80 else "high"
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                    "status": "normal" if memory.percent < 85 else "high"
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent": (disk.used / disk.total) * 100,
                    "status": "normal" if (disk.used / disk.total) * 100 < 90 else "high"
                },
                "processes": process_count,
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                "alerts_count": len([a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(hours=1)])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return {"error": str(e)}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "virtual": {
                    "total": memory.total,
                    "used": memory.used,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_alerts(self, severity: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert["severity"] == severity
            ]
        
        return sorted(filtered_alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List]:
        """Get performance trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data
        recent_cpu = [(ts, val) for ts, val in self.cpu_history if ts > cutoff_time]
        recent_memory = [(ts, val) for ts, val in self.memory_history if ts > cutoff_time]
        recent_disk = [(ts, val) for ts, val in self.disk_history if ts > cutoff_time]
        
        return {
            "cpu": recent_cpu,
            "memory": recent_memory,
            "disk": recent_disk
        }






