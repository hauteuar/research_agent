# utils/health_monitor.py
"""
System Health Monitor for Opulence Deep Research Mainframe Agent
Monitors system resources, performance metrics, and component health
"""

import psutil
import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class SystemAlert:
    """System alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    component: str
    timestamp: datetime
    threshold_value: float
    current_value: float
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_metrics: Dict[int, Dict[str, Any]]
    network_io: Dict[str, int]
    process_count: int
    load_average: Optional[List[float]] = None

class HealthMonitor:
    """Comprehensive system health monitoring for GPU-accelerated workloads"""
    
    def __init__(self, history_size: int = 1000, alert_retention_hours: int = 24):
        self.logger = logging.getLogger(__name__)
        self.history_size = history_size
        self.alert_retention_hours = alert_retention_hours
        
        # Performance history storage
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        
        # Alerts storage
        self.alerts = deque(maxlen=1000)
        self.alert_counter = 0
        
        # Configurable thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 85, "critical": 95},
            "disk_percent": {"warning": 90, "critical": 98},
            "gpu_memory_percent": {"warning": 85, "critical": 95},
            "gpu_temperature": {"warning": 80, "critical": 90},
            "response_time": {"warning": 30, "critical": 60},
            "process_count": {"warning": 500, "critical": 1000}
        }
        
        # Component health tracking
        self.component_health = {
            "llm_engines": {},
            "agents": {},
            "databases": {},
            "network": "healthy"
        }
        
        # Threading
        self.monitoring_active = True
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # Initialize GPU monitoring if available
        self.gpu_available = self._init_gpu_monitoring()
        
        # Start monitoring
        self.start_monitoring()
    
    def _init_gpu_monitoring(self) -> bool:
        """Initialize GPU monitoring capabilities"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available. GPU monitoring disabled.")
            return False
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. GPU monitoring disabled.")
            return False
        
        try:
            if NVML_AVAILABLE:
                nvml.nvmlInit()
                self.logger.info("NVIDIA ML monitoring initialized")
            
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"GPU monitoring initialized for {gpu_count} GPUs")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU monitoring: {str(e)}")
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_system(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store in history
                with self.lock:
                    self.cpu_history.append((metrics.timestamp, metrics.cpu_percent))
                    self.memory_history.append((metrics.timestamp, metrics.memory_percent))
                    self.disk_history.append((metrics.timestamp, metrics.disk_percent))
                    
                    if metrics.gpu_metrics:
                        self.gpu_history.append((metrics.timestamp, metrics.gpu_metrics))
                    
                    if metrics.network_io:
                        self.network_history.append((metrics.timestamp, metrics.network_io))
                
                # Check thresholds and generate alerts
                self._check_thresholds(metrics)
                
                # Clean old alerts
                self._cleanup_old_alerts()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                time.sleep(60)  # Continue monitoring even on errors
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix/Linux only)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            pass  # Windows doesn't have load average
        
        # GPU metrics
        gpu_metrics = {}
        if self.gpu_available:
            gpu_metrics = self._collect_gpu_metrics()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            gpu_metrics=gpu_metrics,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average
        )
    
    def _collect_gpu_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Collect GPU-specific metrics"""
        gpu_metrics = {}
        
        try:
            gpu_count = torch.cuda.device_count()
            
            for gpu_id in range(gpu_count):
                metrics = {
                    "gpu_id": gpu_id,
                    "memory_allocated": 0,
                    "memory_cached": 0,
                    "memory_total": 0,
                    "memory_percent": 0,
                    "temperature": 0,
                    "utilization": 0,
                    "power_usage": 0,
                    "name": "Unknown"
                }
                
                # PyTorch memory info
                try:
                    with torch.cuda.device(gpu_id):
                        memory_allocated = torch.cuda.memory_allocated(gpu_id)
                        memory_cached = torch.cuda.memory_reserved(gpu_id)
                        
                        # Get GPU properties
                        props = torch.cuda.get_device_properties(gpu_id)
                        memory_total = props.total_memory
                        
                        metrics.update({
                            "memory_allocated": memory_allocated,
                            "memory_cached": memory_cached,
                            "memory_total": memory_total,
                            "memory_percent": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                            "name": props.name
                        })
                except Exception as e:
                    self.logger.debug(f"Failed to get PyTorch metrics for GPU {gpu_id}: {str(e)}")
                
                # NVIDIA ML metrics (if available)
                if NVML_AVAILABLE:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        
                        # Temperature
                        try:
                            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                            metrics["temperature"] = temp
                        except:
                            pass
                        
                        # Utilization
                        try:
                            util = nvml.nvmlDeviceGetUtilizationRates(handle)
                            metrics["utilization"] = util.gpu
                        except:
                            pass
                        
                        # Power usage
                        try:
                            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                            metrics["power_usage"] = power
                        except:
                            pass
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to get NVML metrics for GPU {gpu_id}: {str(e)}")
                
                gpu_metrics[gpu_id] = metrics
        
        except Exception as e:
            self.logger.error(f"Failed to collect GPU metrics: {str(e)}")
        
        return gpu_metrics
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check metrics against thresholds and generate alerts"""
        # CPU threshold check
        self._check_metric_threshold(
            "cpu_percent", metrics.cpu_percent, "system", "CPU usage"
        )
        
        # Memory threshold check
        self._check_metric_threshold(
            "memory_percent", metrics.memory_percent, "system", "Memory usage"
        )
        
        # Disk threshold check
        self._check_metric_threshold(
            "disk_percent", metrics.disk_percent, "system", "Disk usage"
        )
        
        # Process count check
        self._check_metric_threshold(
            "process_count", metrics.process_count, "system", "Process count"
        )
        
        # GPU threshold checks
        for gpu_id, gpu_data in metrics.gpu_metrics.items():
            self._check_metric_threshold(
                "gpu_memory_percent", gpu_data["memory_percent"], 
                f"gpu_{gpu_id}", f"GPU {gpu_id} memory usage"
            )
            
            if gpu_data["temperature"] > 0:
                self._check_metric_threshold(
                    "gpu_temperature", gpu_data["temperature"],
                    f"gpu_{gpu_id}", f"GPU {gpu_id} temperature"
                )
    
    def _check_metric_threshold(self, metric_name: str, current_value: float, 
                              component: str, description: str):
        """Check individual metric against thresholds"""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        severity = None
        threshold_value = None
        
        if current_value >= thresholds.get("critical", float('inf')):
            severity = "critical"
            threshold_value = thresholds["critical"]
        elif current_value >= thresholds.get("warning", float('inf')):
            severity = "warning" 
            threshold_value = thresholds["warning"]
        
        if severity:
            self._create_alert(
                alert_type=metric_name,
                severity=severity,
                message=f"{description} is {severity}: {current_value:.1f}%",
                component=component,
                threshold_value=threshold_value,
                current_value=current_value
            )
    
    def _create_alert(self, alert_type: str, severity: str, message: str, 
                     component: str, threshold_value: float, current_value: float):
        """Create and store a new alert"""
        with self.lock:
            self.alert_counter += 1
            
            alert = SystemAlert(
                alert_id=f"ALERT_{self.alert_counter:06d}",
                alert_type=alert_type,
                severity=severity,
                message=message,
                component=component,
                timestamp=datetime.now(),
                threshold_value=threshold_value,
                current_value=current_value
            )
            
            self.alerts.append(alert)
            
            # Log the alert
            log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
            self.logger.log(log_level, f"ALERT [{severity.upper()}]: {message}")
    
    def _cleanup_old_alerts(self):
        """Remove old alerts beyond retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        with self.lock:
            # Filter out old alerts
            self.alerts = deque(
                [alert for alert in self.alerts if alert.timestamp > cutoff_time],
                maxlen=self.alerts.maxlen
            )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Current metrics
            current_metrics = self._collect_metrics()
            
            # Recent alerts
            recent_alerts = self.get_alerts(hours=1)
            critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
            warning_alerts = [a for a in recent_alerts if a.severity == "warning"]
            
            # Overall health status
            if critical_alerts:
                overall_status = "critical"
            elif warning_alerts:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            return {
                "overall_status": overall_status,
                "timestamp": current_metrics.timestamp.isoformat(),
                "cpu": {
                    "percent": current_metrics.cpu_percent,
                    "status": self._get_metric_status("cpu_percent", current_metrics.cpu_percent)
                },
                "memory": {
                    "percent": current_metrics.memory_percent,
                    "total_gb": psutil.virtual_memory().total / (1024**3),
                    "available_gb": psutil.virtual_memory().available / (1024**3),
                    "status": self._get_metric_status("memory_percent", current_metrics.memory_percent)
                },
                "disk": {
                    "percent": current_metrics.disk_percent,
                    "total_gb": psutil.disk_usage('/').total / (1024**3),
                    "free_gb": psutil.disk_usage('/').free / (1024**3),
                    "status": self._get_metric_status("disk_percent", current_metrics.disk_percent)
                },
                "gpu": self._format_gpu_status(current_metrics.gpu_metrics),
                "processes": current_metrics.process_count,
                "network": current_metrics.network_io,
                "load_average": current_metrics.load_average,
                "alerts": {
                    "critical": len(critical_alerts),
                    "warning": len(warning_alerts),
                    "total_recent": len(recent_alerts)
                },
                "component_health": self.component_health
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status string for a metric value"""
        if metric_name not in self.thresholds:
            return "unknown"
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get("critical", float('inf')):
            return "critical"
        elif value >= thresholds.get("warning", float('inf')):
            return "warning"
        else:
            return "normal"
    
    def _format_gpu_status(self, gpu_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Format GPU metrics for status response"""
        if not gpu_metrics:
            return {"available": False, "count": 0}
        
        gpu_status = {
            "available": True,
            "count": len(gpu_metrics),
            "devices": {}
        }
        
        for gpu_id, metrics in gpu_metrics.items():
            gpu_status["devices"][f"gpu_{gpu_id}"] = {
                "name": metrics.get("name", "Unknown"),
                "memory_percent": metrics.get("memory_percent", 0),
                "memory_total_gb": metrics.get("memory_total", 0) / (1024**3),
                "memory_used_gb": metrics.get("memory_allocated", 0) / (1024**3),
                "temperature": metrics.get("temperature", 0),
                "utilization": metrics.get("utilization", 0),
                "power_usage": metrics.get("power_usage", 0),
                "status": self._get_metric_status("gpu_memory_percent", metrics.get("memory_percent", 0))
            }
        
        return gpu_status
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            memory_info = {
                "virtual": {
                    "total": virtual_memory.total,
                    "available": virtual_memory.available,
                    "used": virtual_memory.used,
                    "percent": virtual_memory.percent,
                    "free": virtual_memory.free,
                    "buffers": getattr(virtual_memory, 'buffers', 0),
                    "cached": getattr(virtual_memory, 'cached', 0)
                },
                "swap": {
                    "total": swap_memory.total,
                    "used": swap_memory.used,
                    "free": swap_memory.free,
                    "percent": swap_memory.percent
                }
            }
            
            # Add GPU memory if available
            if self.gpu_available:
                gpu_memory = {}
                gpu_metrics = self._collect_gpu_metrics()
                
                for gpu_id, metrics in gpu_metrics.items():
                    gpu_memory[f"gpu_{gpu_id}"] = {
                        "allocated": metrics.get("memory_allocated", 0),
                        "cached": metrics.get("memory_cached", 0),
                        "total": metrics.get("memory_total", 0),
                        "percent": metrics.get("memory_percent", 0)
                    }
                
                memory_info["gpu"] = gpu_memory
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {str(e)}")
            return {"error": str(e)}
    
    def get_alerts(self, severity: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            filtered_alerts = [
                alert for alert in self.alerts
                if alert.timestamp > cutoff_time and not alert.resolved
            ]
            
            if severity:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert.severity == severity
                ]
            
            # Convert to dictionary format
            alert_dicts = []
            for alert in sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True):
                alert_dicts.append({
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat(),
                    "threshold_value": alert.threshold_value,
                    "current_value": alert.current_value,
                    "resolved": alert.resolved
                })
            
            return alert_dicts
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List]:
        """Get performance trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Filter recent data
            recent_cpu = [(ts.isoformat(), val) for ts, val in self.cpu_history if ts > cutoff_time]
            recent_memory = [(ts.isoformat(), val) for ts, val in self.memory_history if ts > cutoff_time]
            recent_disk = [(ts.isoformat(), val) for ts, val in self.disk_history if ts > cutoff_time]
            recent_gpu = [(ts.isoformat(), val) for ts, val in self.gpu_history if ts > cutoff_time]
            recent_network = [(ts.isoformat(), val) for ts, val in self.network_history if ts > cutoff_time]
            
            return {
                "cpu": recent_cpu,
                "memory": recent_memory,
                "disk": recent_disk,
                "gpu": recent_gpu,
                "network": recent_network,
                "timeframe_hours": hours,
                "data_points": len(recent_cpu)
            }
    
    def update_component_health(self, component_type: str, component_name: str, 
                              status: str, details: Dict[str, Any] = None):
        """Update health status for a specific component"""
        with self.lock:
            if component_type not in self.component_health:
                self.component_health[component_type] = {}
            
            self.component_health[component_type][component_name] = {
                "status": status,
                "last_updated": datetime.now().isoformat(),
                "details": details or {}
            }
            
            self.logger.info(f"Component health updated: {component_type}.{component_name} = {status}")
    
    def set_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]):
        """Update monitoring thresholds"""
        with self.lock:
            for metric, thresholds in new_thresholds.items():
                if metric in self.thresholds:
                    self.thresholds[metric].update(thresholds)
                else:
                    self.thresholds[metric] = thresholds
            
            self.logger.info("Monitoring thresholds updated")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        status = self.get_current_status()
        alerts = self.get_alerts(hours=24)
        trends = self.get_performance_trends(hours=6)  # Last 6 hours
        
        return {
            "summary": {
                "overall_status": status["overall_status"],
                "monitoring_active": self.monitoring_active,
                "gpu_available": self.gpu_available,
                "uptime_hours": self._get_system_uptime_hours()
            },
            "current_status": status,
            "recent_alerts": alerts[:10],  # Last 10 alerts
            "performance_trends": {
                "cpu_avg": self._calculate_average(trends["cpu"]),
                "memory_avg": self._calculate_average(trends["memory"]),
                "disk_avg": self._calculate_average(trends["disk"])
            },
            "thresholds": self.thresholds
        }
    
    def _get_system_uptime_hours(self) -> float:
        """Get system uptime in hours"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            return uptime_seconds / 3600
        except:
            return 0.0
    
    def _calculate_average(self, time_series_data: List[Tuple]) -> float:
        """Calculate average from time series data"""
        if not time_series_data:
            return 0.0
        
        values = [val for _, val in time_series_data if isinstance(val, (int, float))]
        return sum(values) / len(values) if values else 0.0