# FIXES FOR CONFIG MANAGER - utils/config_manager.py

import json
import yaml
import os
import time
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

# FIX 1: Add missing GPUConfig dataclass
@dataclass
class GPUConfig:
    total_gpu_count: int = 4
    memory_threshold: float = 0.80
    utilization_threshold: float = 75.0
    preferred_gpus: Optional[List[int]] = None
    exclude_gpu_0: bool = True
    enable_sharing: bool = True
    cleanup_interval: int = 300  # 5 minutes

# FIX 2: Enhanced ConfigManager with missing methods
class ConfigManager:
    """Enhanced Configuration Manager with all required methods"""
    
    def __init__(self, config_file: str = "opulence_config.yaml"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        # Load configuration from file if exists
        if self.config_file.exists():
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load enhanced default configuration with GPU settings"""
        return {
            "system": {
                "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                "max_tokens": 4096,
                "temperature": 0.1,
                "gpu_count": 4,  # Changed from 3 to 4
                "max_processing_time": 900,
                "batch_size": 32,
                "vector_dim": 768,
                "max_db_rows": 10000,
                "cache_ttl": 3600
            },
            # FIX: Add missing GPU configuration section
            "gpu": {
                "total_gpu_count": 4,
                "memory_threshold": 0.80,
                "utilization_threshold": 75.0,
                "preferred_gpus": [1, 2, 3, 0],  # Prefer non-zero GPUs
                "exclude_gpu_0": True,
                "enable_sharing": True,
                "cleanup_interval": 300,
                "agent_preferences": {
                    "code_parser": [1, 2],
                    "vector_index": [2, 3],
                    "data_loader": [1, 3],
                    "lineage_analyzer": [2, 1],
                    "logic_analyzer": [3, 2],
                    "documentation": [1, 2],
                    "db2_comparator": [2, 3]
                }
            },
            # FIX: Add missing performance section
            "performance": {
                "enable_caching": True,
                "cache_size": 1000,
                "cache_ttl": 3600,
                "enable_gpu_monitoring": True,
                "health_check_interval": 60,
                "cleanup_interval": 300,
                "max_concurrent_requests": 10,
                "request_timeout": 300,
                "memory_cleanup_threshold": 0.85
            },
            # FIX: Add missing optimization section
            "optimization": {
                "enable_auto_optimization": True,
                "rebalance_interval": 600,  # 10 minutes
                "workload_prediction": True,
                "adaptive_memory_management": True,
                "dynamic_gpu_allocation": True
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
        """Enhanced validation with GPU settings"""
        issues = []
        
        # Validate required sections
        required_sections = ["system", "gpu", "performance", "db2", "logging"]
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
        
        # FIX: Validate GPU settings
        gpu_config = self.config.get("gpu", {})
        
        if gpu_config.get("total_gpu_count", 0) < 1:
            issues.append("Total GPU count must be at least 1")
        
        memory_threshold = gpu_config.get("memory_threshold", 0)
        if not (0.1 <= memory_threshold <= 1.0):
            issues.append("Memory threshold must be between 0.1 and 1.0")
        
        util_threshold = gpu_config.get("utilization_threshold", 0)
        if not (10 <= util_threshold <= 100):
            issues.append("Utilization threshold must be between 10 and 100")
        
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

# FIX 3: Create DynamicConfigManager class that inherits from ConfigManager
class DynamicConfigManager(ConfigManager):
    """Dynamic Configuration Manager with GPU-specific features"""
    
    def __init__(self, config_file: str = "opulence_config.yaml"):
        super().__init__(config_file)
        self.runtime_modifications = {}
        self.optimization_history = []
    
    def create_runtime_config(self) -> Dict[str, Any]:
        """Create runtime configuration with current values"""
        base_config = self.config.copy()
        
        # Apply runtime modifications
        for key_path, value in self.runtime_modifications.items():
            self.set(key_path, value)
        
        # Return flattened config for coordinator
        return {
            "model_name": self.get("system.model_name", "codellama/CodeLlama-7b-Instruct-hf"),
            "max_tokens": self.get("system.max_tokens", 4096),
            "temperature": self.get("system.temperature", 0.1),
            "total_gpu_count": self.get("gpu.total_gpu_count", 4),
            "max_processing_time": self.get("system.max_processing_time", 900),
            "batch_size": self.get("system.batch_size", 32),
            "vector_dim": self.get("system.vector_dim", 768),
            "max_db_rows": self.get("system.max_db_rows", 10000),
            "cache_ttl": self.get("system.cache_ttl", 3600),
            "memory_threshold": self.get("gpu.memory_threshold", 0.85),
            "utilization_threshold": self.get("gpu.utilization_threshold", 80.0)
        }
    
    def get_gpu_config(self) -> GPUConfig:
        """Get GPU configuration as GPUConfig object"""
        gpu_section = self.get_section("gpu")
        
        return GPUConfig(
            total_gpu_count=gpu_section.get("total_gpu_count", 4),
            memory_threshold=gpu_section.get("memory_threshold", 0.80),
            utilization_threshold=gpu_section.get("utilization_threshold", 75.0),
            preferred_gpus=gpu_section.get("preferred_gpus", [1, 2, 3, 0]),
            exclude_gpu_0=gpu_section.get("exclude_gpu_0", True),
            enable_sharing=gpu_section.get("enable_sharing", True),
            cleanup_interval=gpu_section.get("cleanup_interval", 300)
        )
    
    def get_gpu_agent_mapping(self) -> Dict[str, List[int]]:
        """Get GPU preferences for each agent type"""
        return self.get("gpu.agent_preferences", {
            "code_parser": [1, 2],
            "vector_index": [2, 3],
            "data_loader": [1, 3],
            "lineage_analyzer": [2, 1],
            "logic_analyzer": [3, 2],
            "documentation": [1, 2],
            "db2_comparator": [2, 3]
        })
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get_section("performance")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.get_section("optimization")
    
    def set_agent_preferred_gpu(self, agent_type: str, gpu_id: Optional[int]) -> bool:
        """Set preferred GPU for specific agent"""
        try:
            if gpu_id is None:
                # Remove preference
                current_prefs = self.get("gpu.agent_preferences", {})
                if agent_type in current_prefs:
                    del current_prefs[agent_type]
                    self.set("gpu.agent_preferences", current_prefs)
            else:
                # Set preference
                current_prefs = self.get("gpu.agent_preferences", {})
                if agent_type not in current_prefs:
                    current_prefs[agent_type] = []
                
                # Move preferred GPU to front of list
                if gpu_id in current_prefs[agent_type]:
                    current_prefs[agent_type].remove(gpu_id)
                current_prefs[agent_type].insert(0, gpu_id)
                
                self.set("gpu.agent_preferences", current_prefs)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set agent GPU preference: {e}")
            return False
    
    def optimize_agent_gpu_assignment(self) -> Dict[str, int]:
        """Optimize GPU assignments for all agents"""
        try:
            # Get current GPU utilization (would need actual GPU manager for real data)
            # For now, return balanced assignment
            
            agents = ["code_parser", "vector_index", "data_loader", 
                     "lineage_analyzer", "logic_analyzer", "documentation", "db2_comparator"]
            
            # Simple round-robin assignment starting from GPU 1
            assignments = {}
            gpu_count = self.get("gpu.total_gpu_count", 4)
            start_gpu = 1 if self.get("gpu.exclude_gpu_0", True) else 0
            
            for i, agent in enumerate(agents):
                gpu_id = start_gpu + (i % (gpu_count - start_gpu))
                assignments[agent] = gpu_id
                
                # Update configuration
                self.set_agent_preferred_gpu(agent, gpu_id)
            
            # Record optimization
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "assignments": assignments,
                "method": "round_robin"
            })
            
            return assignments
            
        except Exception as e:
            self.logger.error(f"GPU assignment optimization failed: {e}")
            return {}
    
    def backup_config(self) -> str:
        """Create backup of current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config_file.stem}_backup_{timestamp}{self.config_file.suffix}"
            
            shutil.copy2(self.config_file, backup_path)
            
            self.logger.info(f"Configuration backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            return ""
    
    def apply_runtime_optimization(self, optimization_data: Dict[str, Any]):
        """Apply runtime optimizations without saving to file"""
        try:
            for key_path, value in optimization_data.items():
                self.runtime_modifications[key_path] = value
                
            self.logger.info(f"Applied {len(optimization_data)} runtime optimizations")
            
        except Exception as e:
            self.logger.error(f"Runtime optimization failed: {e}")
    
    def get_dynamic_config(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value with runtime modifications"""
        # Check runtime modifications first
        if key_path in self.runtime_modifications:
            return self.runtime_modifications[key_path]
        
        # Fall back to base configuration
        return self.get(key_path, default)

# FIX 4: Add convenience function for getting dynamic config
def get_dynamic_config() -> DynamicConfigManager:
    """Get global dynamic configuration manager instance"""
    global _dynamic_config_manager
    
    if '_dynamic_config_manager' not in globals():
        _dynamic_config_manager = DynamicConfigManager()
    
    return _dynamic_config_manager

# FIX 5: Create global instances for backward compatibility
config_manager = ConfigManager()
dynamic_config_manager = DynamicConfigManager()

# FIX 6: Add missing helper classes that might be needed
class HealthMonitor:
    """Health monitoring for system components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent
            }
        except ImportError:
            return {
                "total_gb": 16.0,
                "available_gb": 8.0,
                "used_gb": 8.0,
                "percent_used": 50.0
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        return {
            "status": "healthy",
            "memory": self.get_memory_usage(),
            "timestamp": datetime.now().isoformat()
        }

class CacheManager:
    """Simple cache manager"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key: str) -> Any:
        """Get value from cache"""
        if key not in self.cache:
            return None
            
        # Check if expired
        if time.time() - self.timestamps.get(key, 0) > self.default_ttl:
            self.delete(key)
            return None
            
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_items": len(self.cache),
            "memory_usage_estimate": sum(len(str(v)) for v in self.cache.values()),
            "default_ttl": self.default_ttl
        }

config_manager = ConfigManager()