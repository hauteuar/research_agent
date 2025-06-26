# utils/dynamic_config_manager.py
"""
Enhanced Configuration Manager for Dynamic GPU System
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import torch
from dataclasses import dataclass

@dataclass
class GPUConfig:
    """GPU-specific configuration"""
    total_gpu_count: int = 4
    memory_threshold: float = 0.85
    utilization_threshold: float = 80.0
    preferred_gpu_order: List[int] = None
    fallback_enabled: bool = True
    max_retries: int = 3
    allocation_timeout: int = 30
    
    def __post_init__(self):
        if self.preferred_gpu_order is None:
            self.preferred_gpu_order = list(range(self.total_gpu_count))

class DynamicConfigManager:
    """Enhanced configuration manager for dynamic GPU system"""
    
    def __init__(self, config_file: str = "dynamic_opulence_config.yaml"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        # Load configuration from file if exists
        if self.config_file.exists():
            self.load_config()
        else:
            # Auto-detect GPU configuration
            self._auto_detect_gpu_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration with dynamic GPU support"""
        return {
            "system": {
                "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                "max_tokens": 4096,
                "temperature": 0.1,
                "max_processing_time": 900,
                "batch_size": 32,
                "vector_dim": 768,
                "max_db_rows": 10000,
                "cache_ttl": 3600,
                "enable_dynamic_gpu": True,
                "auto_detect_gpu": True
            },
            "gpu": {
                "total_gpu_count": 4,
                "memory_threshold": 0.85,
                "utilization_threshold": 80.0,
                "preferred_gpu_order": [0, 1, 2, 3],
                "fallback_enabled": True,
                "max_retries": 3,
                "allocation_timeout": 30,
                "monitoring_interval": 5,
                "cleanup_interval": 300,
                "engine_cache_size": 4,
                "auto_scale_engines": True
            },
            "agents": {
                "code_parser": {
                    "preferred_gpu": None,
                    "memory_requirement": "medium",
                    "priority": "normal"
                },
                "vector_index": {
                    "preferred_gpu": None,
                    "memory_requirement": "high",
                    "priority": "high"
                },
                "data_loader": {
                    "preferred_gpu": None,
                    "memory_requirement": "low",
                    "priority": "normal"
                },
                "lineage_analyzer": {
                    "preferred_gpu": None,
                    "memory_requirement": "medium",
                    "priority": "high"
                },
                "logic_analyzer": {
                    "preferred_gpu": None,
                    "memory_requirement": "medium",
                    "priority": "normal"
                },
                "documentation": {
                    "preferred_gpu": None,
                    "memory_requirement": "low",
                    "priority": "low"
                },
                "db2_comparator": {
                    "preferred_gpu": None,
                    "memory_requirement": "medium",
                    "priority": "normal"
                }
            },
            "db2": {
                "database": "TESTDB",
                "hostname": "localhost",
                "port": "50000",
                "username": "db2user",
                "password": "password",
                "connection_timeout": 30,
                "max_connections": 10,
                "pool_size": 5
            },
            "logging": {
                "level": "INFO",
                "file": "dynamic_opulence.log",
                "max_size_mb": 100,
                "backup_count": 5,
                "gpu_logging": True,
                "performance_logging": True
            },
            "security": {
                "enable_auth": False,
                "session_timeout": 3600,
                "allowed_file_types": [
                    ".cbl", ".cob", ".jcl", ".csv", ".ddl", 
                    ".sql", ".dcl", ".copy", ".cpy", ".zip"
                ],
                "max_file_size_mb": 100,
                "enable_encryption": False
            },
            "performance": {
                "enable_caching": True,
                "cache_size": 1000,
                "enable_gpu_monitoring": True,
                "health_check_interval": 60,
                "cleanup_interval": 300,
                "enable_metrics": True,
                "metrics_retention_days": 7,
                "enable_auto_optimization": True
            }
        }
    
    def _auto_detect_gpu_config(self):
        """Auto-detect GPU configuration"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.config["gpu"]["total_gpu_count"] = gpu_count
                self.config["gpu"]["preferred_gpu_order"] = list(range(gpu_count))
                
                # Get GPU memory info for threshold adjustment
                total_memory = []
                for i in range(gpu_count):
                    with torch.cuda.device(i):
                        props = torch.cuda.get_device_properties(i)
                        total_memory.append(props.total_memory)
                
                # Adjust thresholds based on GPU memory
                min_memory_gb = min(total_memory) / (1024**3)
                if min_memory_gb < 8:
                    self.config["gpu"]["memory_threshold"] = 0.75
                    self.config["gpu"]["utilization_threshold"] = 70.0
                elif min_memory_gb > 24:
                    self.config["gpu"]["memory_threshold"] = 0.90
                    self.config["gpu"]["utilization_threshold"] = 85.0
                
                self.logger.info(f"Auto-detected {gpu_count} GPUs with memory range: "
                               f"{min_memory_gb:.1f}GB - {max(total_memory) / (1024**3):.1f}GB")
                
            else:
                self.logger.warning("CUDA not available, disabling GPU features")
                self.config["system"]["enable_dynamic_gpu"] = False
                self.config["gpu"]["total_gpu_count"] = 0
                
        except Exception as e:
            self.logger.error(f"GPU auto-detection failed: {str(e)}")
            self.config["system"]["enable_dynamic_gpu"] = False
    
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
            
            # Validate GPU configuration
            if self.config["system"].get("auto_detect_gpu", True):
                self._auto_detect_gpu_config()
            
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
    
    def get_gpu_config(self) -> GPUConfig:
        """Get GPU configuration as dataclass"""
        gpu_section = self.config.get("gpu", {})
        return GPUConfig(
            total_gpu_count=gpu_section.get("total_gpu_count", 4),
            memory_threshold=gpu_section.get("memory_threshold", 0.85),
            utilization_threshold=gpu_section.get("utilization_threshold", 80.0),
            preferred_gpu_order=gpu_section.get("preferred_gpu_order", [0, 1, 2, 3]),
            fallback_enabled=gpu_section.get("fallback_enabled", True),
            max_retries=gpu_section.get("max_retries", 3),
            allocation_timeout=gpu_section.get("allocation_timeout", 30)
        )
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        return self.config.get("agents", {}).get(agent_type, {})
    
    def set_agent_preferred_gpu(self, agent_type: str, gpu_id: Optional[int]) -> bool:
        """Set preferred GPU for specific agent"""
        try:
            if "agents" not in self.config:
                self.config["agents"] = {}
            
            if agent_type not in self.config["agents"]:
                self.config["agents"][agent_type] = {}
            
            self.config["agents"][agent_type]["preferred_gpu"] = gpu_id
            self.logger.info(f"Set preferred GPU {gpu_id} for agent {agent_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set preferred GPU for {agent_type}: {str(e)}")
            return False
    
    def get_preferred_gpu_order(self) -> List[int]:
        """Get preferred GPU order"""
        return self.config.get("gpu", {}).get("preferred_gpu_order", [0, 1, 2, 3])
    
    def set_preferred_gpu_order(self, gpu_order: List[int]) -> bool:
        """Set preferred GPU order"""
        try:
            if "gpu" not in self.config:
                self.config["gpu"] = {}
            
            self.config["gpu"]["preferred_gpu_order"] = gpu_order
            self.logger.info(f"Set preferred GPU order: {gpu_order}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set GPU order: {str(e)}")
            return False
    
    def update_gpu_thresholds(self, memory_threshold: float = None, 
                             utilization_threshold: float = None) -> bool:
        """Update GPU utilization thresholds"""
        try:
            if "gpu" not in self.config:
                self.config["gpu"] = {}
            
            if memory_threshold is not None:
                self.config["gpu"]["memory_threshold"] = memory_threshold
                
            if utilization_threshold is not None:
                self.config["gpu"]["utilization_threshold"] = utilization_threshold
            
            self.logger.info(f"Updated GPU thresholds: memory={memory_threshold}, "
                           f"utilization={utilization_threshold}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update GPU thresholds: {str(e)}")
            return False
    
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
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate required sections
        required_sections = ["system", "gpu", "agents", "db2", "logging"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Validate GPU configuration
        gpu_config = self.config.get("gpu", {})
        
        if gpu_config.get("total_gpu_count", 0) < 0:
            issues.append("GPU count cannot be negative")
        
        memory_threshold = gpu_config.get("memory_threshold", 0.85)
        if not 0.1 <= memory_threshold <= 1.0:
            issues.append("Memory threshold must be between 0.1 and 1.0")
        
        utilization_threshold = gpu_config.get("utilization_threshold", 80.0)
        if not 10.0 <= utilization_threshold <= 100.0:
            issues.append("Utilization threshold must be between 10.0 and 100.0")
        
        # Validate preferred GPU order
        preferred_order = gpu_config.get("preferred_gpu_order", [])
        total_gpus = gpu_config.get("total_gpu_count", 0)
        
        if len(preferred_order) != total_gpus:
            issues.append(f"Preferred GPU order length ({len(preferred_order)}) doesn't match GPU count ({total_gpus})")
        
        if len(set(preferred_order)) != len(preferred_order):
            issues.append("Preferred GPU order contains duplicates")
        
        for gpu_id in preferred_order:
            if not isinstance(gpu_id, int) or gpu_id < 0 or gpu_id >= total_gpus:
                issues.append(f"Invalid GPU ID in preferred order: {gpu_id}")
        
        # Validate agent configurations
        agents_config = self.config.get("agents", {})
        valid_agents = ["code_parser", "vector_index", "data_loader", "lineage_analyzer", 
                       "logic_analyzer", "documentation", "db2_comparator"]
        
        for agent_type, agent_config in agents_config.items():
            if agent_type not in valid_agents:
                issues.append(f"Unknown agent type: {agent_type}")
            
            preferred_gpu = agent_config.get("preferred_gpu")
            if preferred_gpu is not None:
                if not isinstance(preferred_gpu, int) or preferred_gpu < 0 or preferred_gpu >= total_gpus:
                    issues.append(f"Invalid preferred GPU for {agent_type}: {preferred_gpu}")
        
        # Validate system settings
        system_config = self.config.get("system", {})
        
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
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get configuration for GPU optimization"""
        return {
            "enable_auto_optimization": self.get("performance.enable_auto_optimization", True),
            "cleanup_interval": self.get("gpu.cleanup_interval", 300),
            "monitoring_interval": self.get("gpu.monitoring_interval", 5),
            "engine_cache_size": self.get("gpu.engine_cache_size", 4),
            "auto_scale_engines": self.get("gpu.auto_scale_engines", True),
            "memory_threshold": self.get("gpu.memory_threshold", 0.85),
            "utilization_threshold": self.get("gpu.utilization_threshold", 80.0)
        }
    
    def create_runtime_config(self) -> Dict[str, Any]:
        """Create runtime configuration for the coordinator"""
        return {
            "model_name": self.get("system.model_name"),
            "max_tokens": self.get("system.max_tokens"),
            "temperature": self.get("system.temperature"),
            "total_gpu_count": self.get("gpu.total_gpu_count"),
            "max_processing_time": self.get("system.max_processing_time"),
            "batch_size": self.get("system.batch_size"),
            "vector_dim": self.get("system.vector_dim"),
            "max_db_rows": self.get("system.max_db_rows"),
            "cache_ttl": self.get("system.cache_ttl"),
            "memory_threshold": self.get("gpu.memory_threshold"),
            "utilization_threshold": self.get("gpu.utilization_threshold"),
            "enable_dynamic_gpu": self.get("system.enable_dynamic_gpu"),
            "gpu_config": self.get_gpu_config()
        }
    
    def export_config(self, export_path: str = None, include_sensitive: bool = False) -> str:
        """Export configuration to string or file"""
        export_config = self.config.copy()
        
        # Remove sensitive information if requested
        if not include_sensitive:
            if "db2" in export_config and "password" in export_config["db2"]:
                export_config["db2"]["password"] = "***HIDDEN***"
        
        config_str = yaml.dump(export_config, default_flow_style=False, indent=2)
        
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    f.write(config_str)
                self.logger.info(f"Configuration exported to {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to export config: {str(e)}")
        
        return config_str
    
    def import_config(self, import_path: str, merge: bool = True) -> bool:
        """Import configuration from file"""
        try:
            with open(import_path, 'r') as f:
                if import_path.endswith('.yaml') or import_path.endswith('.yml'):
                    imported_config = yaml.safe_load(f)
                else:
                    imported_config = json.load(f)
            
            if merge:
                self._merge_config(self.config, imported_config)
            else:
                self.config = imported_config
            
            # Validate imported configuration
            issues = self.validate_config()
            if issues:
                self.logger.warning(f"Configuration issues found after import: {issues}")
                return False
            
            self.logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import config from {import_path}: {str(e)}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._load_default_config()
        self._auto_detect_gpu_config()
        self.logger.info("Configuration reset to defaults")
    
    def backup_config(self, backup_dir: str = "config_backups") -> str:
        """Create a timestamped backup of current configuration"""
        try:
            from datetime import datetime
            
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"opulence_config_backup_{timestamp}.yaml"
            
            self.export_config(str(backup_file), include_sensitive=False)
            
            self.logger.info(f"Configuration backed up to {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Failed to backup config: {str(e)}")
            return ""
    
    def get_gpu_agent_mapping(self) -> Dict[str, Optional[int]]:
        """Get mapping of agents to their preferred GPUs"""
        agents_config = self.config.get("agents", {})
        return {
            agent_type: agent_config.get("preferred_gpu")
            for agent_type, agent_config in agents_config.items()
        }
    
    def optimize_agent_gpu_assignment(self) -> Dict[str, int]:
        """Optimize GPU assignment for agents based on memory requirements"""
        agents_config = self.config.get("agents", {})
        total_gpus = self.get("gpu.total_gpu_count", 4)
        
        # Categorize agents by memory requirements
        high_memory_agents = []
        medium_memory_agents = []
        low_memory_agents = []
        
        for agent_type, config in agents_config.items():
            memory_req = config.get("memory_requirement", "medium")
            if memory_req == "high":
                high_memory_agents.append(agent_type)
            elif memory_req == "medium":
                medium_memory_agents.append(agent_type)
            else:
                low_memory_agents.append(agent_type)
        
        # Assign GPUs to minimize conflicts
        gpu_assignments = {}
        current_gpu = 0
        
        # Assign high memory agents first, spread across GPUs
        for agent in high_memory_agents:
            gpu_assignments[agent] = current_gpu % total_gpus
            current_gpu += 1
        
        # Assign medium memory agents
        for agent in medium_memory_agents:
            gpu_assignments[agent] = current_gpu % total_gpus
            current_gpu += 1
        
        # Assign low memory agents, can share GPUs
        for agent in low_memory_agents:
            gpu_assignments[agent] = current_gpu % total_gpus
            if len(low_memory_agents) > total_gpus:
                current_gpu += 1
        
        # Update configuration
        for agent_type, gpu_id in gpu_assignments.items():
            self.set_agent_preferred_gpu(agent_type, gpu_id)
        
        self.logger.info(f"Optimized GPU assignments: {gpu_assignments}")
        return gpu_assignments
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            "enable_caching": self.get("performance.enable_caching", True),
            "cache_size": self.get("performance.cache_size", 1000),
            "enable_gpu_monitoring": self.get("performance.enable_gpu_monitoring", True),
            "health_check_interval": self.get("performance.health_check_interval", 60),
            "cleanup_interval": self.get("performance.cleanup_interval", 300),
            "enable_metrics": self.get("performance.enable_metrics", True),
            "metrics_retention_days": self.get("performance.metrics_retention_days", 7),
            "cache_ttl": self.get("system.cache_ttl", 3600)
        }


# Global configuration instance
dynamic_config_manager = DynamicConfigManager()

def get_dynamic_config() -> DynamicConfigManager:
    """Get global dynamic configuration manager"""
    return dynamic_config_manager

def get_gpu_config() -> GPUConfig:
    """Get GPU configuration"""
    return dynamic_config_manager.get_gpu_config()

def get_agent_preferred_gpu(agent_type: str) -> Optional[int]:
    """Get preferred GPU for agent"""
    return dynamic_config_manager.get_agent_config(agent_type).get("preferred_gpu")

def set_agent_preferred_gpu(agent_type: str, gpu_id: Optional[int]) -> bool:
    """Set preferred GPU for agent"""
    return dynamic_config_manager.set_agent_preferred_gpu(agent_type, gpu_id)