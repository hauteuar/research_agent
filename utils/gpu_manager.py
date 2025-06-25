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


# utils/cache_manager.py
"""
Cache Manager for efficient data retrieval
"""

import time
import threading
import json
import hashlib
from typing import Any, Optional, Dict
from collections import OrderedDict
import pickle
import logging

class CacheManager:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.ttls = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, key: str) -> str:
        """Generate consistent cache key"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                self.misses += 1
                return None
            
            # Check if expired
            if self._is_expired(cache_key):
                self._remove_key(cache_key)
                self.misses += 1
                return None
            
            # Move to end (mark as recently used)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            self.hits += 1
            
            return value
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            ttl = ttl or self.default_ttl
            
            # Remove existing key if present
            if cache_key in self.cache:
                self._remove_key(cache_key)
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self.cache[cache_key] = value
            self.timestamps[cache_key] = time.time()
            self.ttls[cache_key] = ttl
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key in self.cache:
                self._remove_key(cache_key)
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.logger.info("Cache cleared")
    
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired"""
        if cache_key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[cache_key]
        ttl = self.ttls.get(cache_key, self.default_ttl)
        
        return age > ttl
    
    def _remove_key(self, cache_key: str):
        """Remove key and associated metadata"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.timestamps:
            del self.timestamps[cache_key]
        if cache_key in self.ttls:
            del self.ttls[cache_key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            lru_key = next(iter(self.cache))
            self._remove_key(lru_key)
            self.evictions += 1
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries"""
        while True:
            try:
                with self.lock:
                    expired_keys = [
                        key for key in self.cache.keys()
                        if self._is_expired(key)
                    ]
                    
                    for key in expired_keys:
                        self._remove_key(key)
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                time.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                time.sleep(300)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate cache memory usage in bytes"""
        try:
            total_size = 0
            for key, value in self.cache.items():
                # Rough estimation using pickle
                total_size += len(pickle.dumps(value))
                total_size += len(key.encode())
            
            return total_size
        except:
            return 0
    
    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching a pattern"""
        with self.lock:
            import re
            pattern_regex = re.compile(pattern)
            
            matching_keys = []
            for key in self.cache.keys():
                if pattern_regex.search(key):
                    matching_keys.append(key)
            
            return matching_keys
    
    def cache_info(self) -> str:
        """Get formatted cache information"""
        stats = self.get_stats()
        
        info = f"""
Cache Information:
- Size: {stats['size']}/{stats['max_size']} entries
- Hit Rate: {stats['hit_rate']:.2f}%
- Hits: {stats['hits']}, Misses: {stats['misses']}
- Evictions: {stats['evictions']}
- Memory Usage: {stats['memory_usage_mb']:.2f} MB
        """
        
        return info.strip()


# utils/batch_processor.py
"""
Batch Processing Utilities
"""

import asyncio
import logging
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from pathlib import Path

class BatchProcessor:
    """Handles batch processing of files and operations"""
    
    def __init__(self, max_workers: int = 4, gpu_count: int = 3):
        self.max_workers = max_workers
        self.gpu_count = gpu_count
        self.logger = logging.getLogger(__name__)
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def process_files_batch(self, file_paths: List[Path], 
                                processor_func: Callable, 
                                batch_size: int = 10,
                                use_gpu_distribution: bool = True) -> List[Any]:
        """Process files in batches with GPU distribution"""
        results = []
        
        try:
            # Split files into batches
            batches = [
                file_paths[i:i + batch_size] 
                for i in range(0, len(file_paths), batch_size)
            ]
            
            self.logger.info(f"Processing {len(file_paths)} files in {len(batches)} batches")
            
            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()
                
                # Distribute batch across GPUs if enabled
                if use_gpu_distribution:
                    batch_results = await self._process_batch_with_gpu_distribution(
                        batch, processor_func, batch_idx
                    )
                else:
                    batch_results = await self._process_batch_sequential(
                        batch, processor_func
                    )
                
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.2f}s"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    async def _process_batch_with_gpu_distribution(self, batch: List[Path], 
                                                 processor_func: Callable,
                                                 batch_idx: int) -> List[Any]:
        """Process batch with GPU distribution"""
        tasks = []
        
        for i, file_path in enumerate(batch):
            # Assign GPU based on file index
            gpu_id = i % self.gpu_count
            
            # Create async task
            task = asyncio.create_task(
                self._process_file_on_gpu(file_path, processor_func, gpu_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _process_batch_sequential(self, batch: List[Path], 
                                      processor_func: Callable) -> List[Any]:
        """Process batch sequentially"""
        results = []
        
        for file_path in batch:
            try:
                result = await processor_func(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({"error": str(e), "file": str(file_path)})
        
        return results
    
    async def _process_file_on_gpu(self, file_path: Path, processor_func: Callable, gpu_id: int) -> Any:
        """Process single file on specific GPU"""
        try:
            # Set GPU device context if using CUDA
            import torch
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    result = await processor_func(file_path)
            else:
                result = await processor_func(file_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU {gpu_id} processing failed for {file_path}: {str(e)}")
            return {"error": str(e), "file": str(file_path), "gpu_id": gpu_id}
    
    def process_cpu_intensive_batch(self, items: List[Any], 
                                  processor_func: Callable,
                                  batch_size: int = None) -> List[Any]:
        """Process CPU-intensive tasks using process pool"""
        batch_size = batch_size or self.max_workers
        
        try:
            # Submit tasks to process pool
            futures = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = self.process_pool.submit(self._process_cpu_batch, batch, processor_func)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                batch_results = future.result(timeout=300)  # 5 minute timeout
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"CPU batch processing failed: {str(e)}")
            raise
    
    @staticmethod
    def _process_cpu_batch(batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch in separate process"""
        results = []
        
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "item": str(item)})
        
        return results
    
    def shutdown(self):
        """Shutdown thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.logger.info("Batch processor shutdown complete")


# utils/config_manager.py
"""
Configuration Manager
"""

import json
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class ConfigManager:
    """Manages system configuration"""
    
    def __init__(self, config_file: str = "opulence_config.yaml"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        # Load configuration from file if exists
        if self.config_file.exists():
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "system": {
                "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                "max_tokens": 4096,
                "temperature": 0.1,
                "gpu_count": 3,
                "max_processing_time": 900,
                "batch_size": 32,
                "vector_dim": 768,
                "max_db_rows": 10000,
                "cache_ttl": 3600
            },
            "db2": {
                "database": "TESTDB",
                "hostname": "localhost",
                "port": "50000",
                "username": "db2user",
                "password": "password",
                "connection_timeout": 30
            },
            "logging": {
                "level": "INFO",
                "file": "opulence.log",
                "max_size_mb": 100,
                "backup_count": 5
            },
            "security": {
                "enable_auth": False,
                "session_timeout": 3600,
                "allowed_file_types": [
                    ".cbl", ".cob", ".jcl", ".csv", ".ddl", 
                    ".sql", ".dcl", ".copy", ".cpy", ".zip"
                ]
            },
            "performance": {
                "enable_caching": True,
                "cache_size": 1000,
                "enable_gpu_monitoring": True,
                "health_check_interval": 60,
                "cleanup_interval": 300
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.suffix.lower() == '.yaml':
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
            
            # Merge with default config
            self._merge_config(self.config, file_config)
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_file}: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_file.suffix.lower() == '.yaml':
                with open(self.config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            else:
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_file}: {str(e)}")
            return False
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        try:
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set config {key_path}: {str(e)}")
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section].update(updates)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update section {section}: {str(e)}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate required sections
        required_sections = ["system", "db2", "logging"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Validate system settings
        system_config = self.config.get("system", {})
        
        if system_config.get("gpu_count", 0) < 1:
            issues.append("GPU count must be at least 1")
        
        if system_config.get("max_processing_time", 0) < 60:
            issues.append("Max processing time should be at least 60 seconds")
        
        if system_config.get("batch_size", 0) < 1:
            issues.append("Batch size must be at least 1")
        
        # Validate DB2 settings
        db2_config = self.config.get("db2", {})
        required_db2_fields = ["database", "hostname", "port", "username"]
        
        for field in required_db2_fields:
            if not db2_config.get(field):
                issues.append(f"DB2 {field} is required")
        
        return issues
    
    def export_config(self, export_path: str = None) -> str:
        """Export configuration to string or file"""
        config_str = yaml.dump(self.config, default_flow_style=False, indent=2)
        
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    f.write(config_str)
                self.logger.info(f"Configuration exported to {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to export config: {str(e)}")
        
        return config_str
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._load_default_config()
        self.logger.info("Configuration reset to defaults")


# Global configuration instance
config_manager = ConfigManager()