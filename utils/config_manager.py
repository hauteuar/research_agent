# utils/config_manager.py
"""
Complete Configuration Manager for Opulence System
Handles dynamic configuration, GPU settings, and runtime optimization
"""

import json
import yaml
import os
import time
import shutil
import sqlite3
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from threading import Lock
from enum import Enum

# Initialize logger
logger = logging.getLogger(__name__)

class ConfigValidationLevel(Enum):
    """Configuration validation levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class GPUConfig:
    """GPU Configuration dataclass"""
    total_gpu_count: int = 4
    memory_threshold: float = 0.80
    utilization_threshold: float = 75.0
    preferred_gpus: Optional[List[int]] = None
    exclude_gpu_0: bool = True
    enable_sharing: bool = True
    cleanup_interval: int = 300  # 5 minutes
    agent_preferences: Optional[Dict[str, List[int]]] = None
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.preferred_gpus is None:
            self.preferred_gpus = [1, 2, 3, 0] if self.total_gpu_count >= 4 else list(range(self.total_gpu_count))
        
        if self.agent_preferences is None:
            self.agent_preferences = {
                "code_parser": [1, 2],
                "vector_index": [2, 3],
                "data_loader": [1, 3],
                "lineage_analyzer": [2, 1],
                "logic_analyzer": [3, 2],
                "documentation": [1, 2],
                "db2_comparator": [2, 3]
            }

@dataclass
class SystemConfig:
    """System Configuration dataclass"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 4096
    temperature: float = 0.1
    gpu_count: int = 4
    max_processing_time: int = 900
    batch_size: int = 32
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600

@dataclass
class PerformanceConfig:
    """Performance Configuration dataclass"""
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    enable_gpu_monitoring: bool = True
    health_check_interval: int = 60
    cleanup_interval: int = 300
    max_concurrent_requests: int = 10
    request_timeout: int = 300
    memory_cleanup_threshold: float = 0.85

@dataclass
class OptimizationConfig:
    """Optimization Configuration dataclass"""
    enable_auto_optimization: bool = True
    rebalance_interval: int = 600  # 10 minutes
    workload_prediction: bool = True
    adaptive_memory_management: bool = True
    dynamic_gpu_allocation: bool = True
    performance_monitoring: bool = True

@dataclass
class DB2Config:
    """DB2 Configuration dataclass"""
    database: str = "TESTDB"
    hostname: str = "localhost"
    port: str = "50000"
    username: str = "db2user"
    password: str = "password"
    connection_timeout: int = 30
    max_connections: int = 10

@dataclass
class SecurityConfig:
    """Security Configuration dataclass"""
    enable_auth: bool = False
    session_timeout: int = 3600
    allowed_file_types: List[str] = None
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = [
                ".cbl", ".cob", ".jcl", ".csv", ".ddl", 
                ".sql", ".dcl", ".copy", ".cpy", ".zip"
            ]

@dataclass
class LoggingConfig:
    """Logging Configuration dataclass"""
    level: str = "INFO"
    file: str = "opulence.log"
    max_size_mb: int = 100
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConfigManager:
    """Enhanced Configuration Manager with comprehensive settings management"""
    
    def __init__(self, config_file: str = "opulence_config.yaml"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        # Configuration sections
        self._config = {}
        self._runtime_modifications = {}
        self._validation_cache = {}
        self._last_loaded = 0
        self._auto_reload = True
        
        # Load or create default configuration
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration with defaults or load from file"""
        try:
            # Create default configuration
            self._config = self._create_default_config()
            
            # Load from file if exists
            if self.config_file.exists():
                self.load_config()
            else:
                # Save default configuration
                self.save_config()
                self.logger.info(f"Created default configuration at {self.config_file}")
            
            self._last_loaded = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration: {e}")
            # Ensure we have basic configuration
            self._config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create comprehensive default configuration"""
        return {
            "system": asdict(SystemConfig()),
            "gpu": asdict(GPUConfig()),
            "performance": asdict(PerformanceConfig()),
            "optimization": asdict(OptimizationConfig()),
            "db2": asdict(DB2Config()),
            "security": asdict(SecurityConfig()),
            "logging": asdict(LoggingConfig()),
            "metadata": {
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "environment": "development"
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file with comprehensive error handling"""
        with self._lock:
            try:
                if not self.config_file.exists():
                    self.logger.warning(f"Configuration file {self.config_file} does not exist")
                    return False
                
                # Determine file format
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == '.json':
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                else:
                    self.logger.error(f"Unsupported configuration file format: {self.config_file.suffix}")
                    return False
                
                if not isinstance(file_config, dict):
                    self.logger.error("Configuration file does not contain a valid dictionary")
                    return False
                
                # Backup current config before merging
                config_backup = self._config.copy()
                
                try:
                    # Merge configurations
                    self._merge_config(self._config, file_config)
                    
                    # Validate merged configuration
                    validation_issues = self.validate_config()
                    
                    # Check for critical validation errors
                    critical_errors = [issue for issue in validation_issues if "error" in issue.lower()]
                    if critical_errors:
                        # Restore backup on critical errors
                        self._config = config_backup
                        self.logger.error(f"Critical configuration errors found, reverting: {critical_errors}")
                        return False
                    
                    # Update metadata
                    self._config["metadata"]["last_modified"] = datetime.now().isoformat()
                    self._last_loaded = time.time()
                    
                    self.logger.info(f"Configuration loaded successfully from {self.config_file}")
                    
                    if validation_issues:
                        self.logger.warning(f"Configuration warnings: {validation_issues}")
                    
                    return True
                    
                except Exception as merge_error:
                    # Restore backup on merge failure
                    self._config = config_backup
                    self.logger.error(f"Failed to merge configuration: {merge_error}")
                    return False
                
            except yaml.YAMLError as e:
                self.logger.error(f"YAML parsing error in {self.config_file}: {e}")
                return False
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error in {self.config_file}: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {self.config_file}: {e}")
                return False
    
    def save_config(self) -> bool:
        """Save current configuration to file with backup"""
        with self._lock:
            try:
                # Create backup if file exists
                if self.config_file.exists():
                    backup_path = self._create_backup()
                    if backup_path:
                        self.logger.debug(f"Configuration backed up to {backup_path}")
                
                # Ensure directory exists
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Update metadata before saving
                self._config["metadata"]["last_modified"] = datetime.now().isoformat()
                
                # Save configuration
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(self._config, f, default_flow_style=False, indent=2, sort_keys=False)
                else:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(self._config, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Configuration saved to {self.config_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration to {self.config_file}: {e}")
                return False
    
    def _create_backup(self) -> Optional[str]:
        """Create backup of current configuration file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self.config_file.stem}_backup_{timestamp}{self.config_file.suffix}"
            backup_path = self.config_file.parent / backup_name
            
            shutil.copy2(self.config_file, backup_path)
            return str(backup_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to create configuration backup: {e}")
            return None
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries with type checking"""
        for key, value in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._merge_config(base[key], value)
                else:
                    # Type validation before assignment
                    if self._validate_config_type(key, value, base[key]):
                        base[key] = value
                    else:
                        self.logger.warning(f"Type mismatch for {key}: expected {type(base[key])}, got {type(value)}")
            else:
                base[key] = value
    
    def _validate_config_type(self, key: str, new_value: Any, existing_value: Any) -> bool:
        """Validate configuration value type compatibility"""
        try:
            # Allow None values
            if new_value is None:
                return True
            
            # Basic type checking
            if type(new_value) == type(existing_value):
                return True
            
            # Allow numeric conversions
            if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                return True
            
            # Allow string to numeric conversion for specific keys
            if isinstance(existing_value, (int, float)) and isinstance(new_value, str):
                try:
                    if isinstance(existing_value, int):
                        int(new_value)
                    else:
                        float(new_value)
                    return True
                except ValueError:
                    return False
            
            return False
            
        except Exception:
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with runtime modifications"""
        with self._lock:
            # Check runtime modifications first
            if key_path in self._runtime_modifications:
                return self._runtime_modifications[key_path]
            
            # Navigate through configuration
            keys = key_path.split('.')
            value = self._config
            
            try:
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default
                return value
            except (KeyError, TypeError, AttributeError):
                return default
    
    def set(self, key_path: str, value: Any, persistent: bool = True) -> bool:
        """Set configuration value using dot notation"""
        with self._lock:
            try:
                if not persistent:
                    # Store as runtime modification
                    self._runtime_modifications[key_path] = value
                    return True
                
                # Navigate to parent of target key
                keys = key_path.split('.')
                config = self._config
                
                for key in keys[:-1]:
                    if key not in config:
                        config[key] = {}
                    elif not isinstance(config[key], dict):
                        # Cannot navigate further
                        self.logger.error(f"Cannot set {key_path}: {key} is not a dictionary")
                        return False
                    config = config[key]
                
                # Validate the value if possible
                final_key = keys[-1]
                if final_key in config:
                    if not self._validate_config_type(final_key, value, config[final_key]):
                        self.logger.warning(f"Type validation failed for {key_path}")
                
                # Set the value
                config[final_key] = value
                
                # Update metadata
                if "metadata" in self._config:
                    self._config["metadata"]["last_modified"] = datetime.now().isoformat()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set config {key_path}: {e}")
                return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        with self._lock:
            return self._config.get(section, {}).copy()
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section with validation"""
        with self._lock:
            try:
                if section not in self._config:
                    self._config[section] = {}
                
                # Validate updates
                for key, value in updates.items():
                    if key in self._config[section]:
                        if not self._validate_config_type(key, value, self._config[section][key]):
                            self.logger.warning(f"Type validation failed for {section}.{key}")
                
                # Apply updates
                self._config[section].update(updates)
                
                # Update metadata
                if "metadata" in self._config:
                    self._config["metadata"]["last_modified"] = datetime.now().isoformat()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update section {section}: {e}")
                return False
    
    def validate_config(self) -> List[str]:
        """Enhanced configuration validation with detailed reporting"""
        issues = []
        
        try:
            # Validate required sections
            required_sections = ["system", "gpu", "performance", "db2", "logging"]
            for section in required_sections:
                if section not in self._config:
                    issues.append(f"ERROR: Missing required section: {section}")
            
            # Validate system settings
            system_config = self._config.get("system", {})
            issues.extend(self._validate_system_config(system_config))
            
            # Validate GPU settings
            gpu_config = self._config.get("gpu", {})
            issues.extend(self._validate_gpu_config(gpu_config))
            
            # Validate performance settings
            perf_config = self._config.get("performance", {})
            issues.extend(self._validate_performance_config(perf_config))
            
            # Validate DB2 settings
            db2_config = self._config.get("db2", {})
            issues.extend(self._validate_db2_config(db2_config))
            
            # Validate security settings
            security_config = self._config.get("security", {})
            issues.extend(self._validate_security_config(security_config))
            
            # Cache validation results
            self._validation_cache = {
                "timestamp": time.time(),
                "issues": issues
            }
            
        except Exception as e:
            issues.append(f"ERROR: Validation process failed: {e}")
        
        return issues
    
    def _validate_system_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate system configuration"""
        issues = []
        
        # GPU count validation
        gpu_count = config.get("gpu_count", 0)
        if not isinstance(gpu_count, int) or gpu_count < 1:
            issues.append("ERROR: GPU count must be a positive integer")
        elif gpu_count > 8:
            issues.append("WARNING: GPU count > 8 may indicate configuration error")
        
        # Processing time validation
        max_time = config.get("max_processing_time", 0)
        if not isinstance(max_time, int) or max_time < 60:
            issues.append("ERROR: Max processing time should be at least 60 seconds")
        elif max_time > 3600:
            issues.append("WARNING: Max processing time > 1 hour may cause timeouts")
        
        # Batch size validation
        batch_size = config.get("batch_size", 0)
        if not isinstance(batch_size, int) or batch_size < 1:
            issues.append("ERROR: Batch size must be a positive integer")
        elif batch_size > 128:
            issues.append("WARNING: Large batch size may cause memory issues")
        
        # Temperature validation
        temperature = config.get("temperature", 0)
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
            issues.append("ERROR: Temperature must be between 0.0 and 2.0")
        
        # Token validation
        max_tokens = config.get("max_tokens", 0)
        if not isinstance(max_tokens, int) or max_tokens < 100:
            issues.append("ERROR: Max tokens must be at least 100")
        elif max_tokens > 32768:
            issues.append("WARNING: Max tokens > 32K may cause memory issues")
        
        return issues
    
    def _validate_gpu_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate GPU configuration"""
        issues = []
        
        # Total GPU count
        total_gpus = config.get("total_gpu_count", 0)
        if not isinstance(total_gpus, int) or total_gpus < 1:
            issues.append("ERROR: Total GPU count must be a positive integer")
        
        # Memory threshold validation
        memory_threshold = config.get("memory_threshold", 0)
        if not isinstance(memory_threshold, (int, float)) or not (0.1 <= memory_threshold <= 1.0):
            issues.append("ERROR: Memory threshold must be between 0.1 and 1.0")
        elif memory_threshold > 0.95:
            issues.append("WARNING: Very high memory threshold may cause OOM errors")
        
        # Utilization threshold validation
        util_threshold = config.get("utilization_threshold", 0)
        if not isinstance(util_threshold, (int, float)) or not (10 <= util_threshold <= 100):
            issues.append("ERROR: Utilization threshold must be between 10 and 100")
        
        # Preferred GPUs validation
        preferred_gpus = config.get("preferred_gpus", [])
        if preferred_gpus is not None:
            if not isinstance(preferred_gpus, list):
                issues.append("ERROR: Preferred GPUs must be a list")
            elif any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in preferred_gpus):
                issues.append("ERROR: Preferred GPU IDs must be non-negative integers")
            elif max(preferred_gpus, default=-1) >= total_gpus:
                issues.append("ERROR: Preferred GPU ID exceeds total GPU count")
        
        # Agent preferences validation
        agent_prefs = config.get("agent_preferences", {})
        if agent_prefs and not isinstance(agent_prefs, dict):
            issues.append("ERROR: Agent preferences must be a dictionary")
        
        return issues
    
    def _validate_performance_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate performance configuration"""
        issues = []
        
        # Cache size validation
        cache_size = config.get("cache_size", 0)
        if not isinstance(cache_size, int) or cache_size < 10:
            issues.append("ERROR: Cache size must be at least 10")
        elif cache_size > 10000:
            issues.append("WARNING: Large cache size may consume excessive memory")
        
        # Timeout validation
        timeout = config.get("request_timeout", 0)
        if not isinstance(timeout, int) or timeout < 30:
            issues.append("ERROR: Request timeout must be at least 30 seconds")
        
        # Concurrent requests validation
        max_concurrent = config.get("max_concurrent_requests", 0)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            issues.append("ERROR: Max concurrent requests must be positive")
        elif max_concurrent > 100:
            issues.append("WARNING: High concurrent request limit may overload system")
        
        return issues
    
    def _validate_db2_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate DB2 configuration"""
        issues = []
        
        required_fields = ["database", "hostname", "port", "username"]
        
        for field in required_fields:
            value = config.get(field)
            if not value or not isinstance(value, str):
                issues.append(f"ERROR: DB2 {field} is required and must be a string")
        
        # Port validation
        port = config.get("port")
        if port:
            try:
                port_num = int(port)
                if not (1 <= port_num <= 65535):
                    issues.append("ERROR: DB2 port must be between 1 and 65535")
            except (ValueError, TypeError):
                issues.append("ERROR: DB2 port must be a valid number")
        
        # Connection timeout validation
        timeout = config.get("connection_timeout", 30)
        if not isinstance(timeout, int) or timeout < 5:
            issues.append("ERROR: DB2 connection timeout must be at least 5 seconds")
        
        return issues
    
    def _validate_security_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate security configuration"""
        issues = []
        
        # File types validation
        allowed_types = config.get("allowed_file_types", [])
        if not isinstance(allowed_types, list):
            issues.append("ERROR: Allowed file types must be a list")
        elif not allowed_types:
            issues.append("WARNING: No allowed file types specified")
        
        # Session timeout validation
        session_timeout = config.get("session_timeout", 3600)
        if not isinstance(session_timeout, int) or session_timeout < 300:
            issues.append("WARNING: Session timeout should be at least 5 minutes")
        
        return issues
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration as SystemConfig object"""
        system_dict = self.get_section("system")
        return SystemConfig(**{k: v for k, v in system_dict.items() if k in SystemConfig.__dataclass_fields__})
    
    def get_gpu_config(self) -> GPUConfig:
        """Get GPU configuration as GPUConfig object"""
        gpu_dict = self.get_section("gpu")
        return GPUConfig(**{k: v for k, v in gpu_dict.items() if k in GPUConfig.__dataclass_fields__})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get_section("performance")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.get_section("optimization")
    
    def get_db2_config(self) -> DB2Config:
        """Get DB2 configuration as DB2Config object"""
        db2_dict = self.get_section("db2")
        return DB2Config(**{k: v for k, v in db2_dict.items() if k in DB2Config.__dataclass_fields__})
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration as SecurityConfig object"""
        security_dict = self.get_section("security")
        return SecurityConfig(**{k: v for k, v in security_dict.items() if k in SecurityConfig.__dataclass_fields__})
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration as LoggingConfig object"""
        logging_dict = self.get_section("logging")
        return LoggingConfig(**{k: v for k, v in logging_dict.items() if k in LoggingConfig.__dataclass_fields__})
    
    def get_gpu_agent_mapping(self) -> Dict[str, List[int]]:
        """Get GPU preferences for each agent type"""
        gpu_config = self.get_gpu_config()
        return gpu_config.agent_preferences or {}
    
    def set_agent_preferred_gpu(self, agent_type: str, gpu_id: Optional[int]) -> bool:
        """Set preferred GPU for specific agent"""
        try:
            current_prefs = self.get("gpu.agent_preferences", {})
            
            if gpu_id is None:
                # Remove preference
                if agent_type in current_prefs:
                    del current_prefs[agent_type]
                    self.set("gpu.agent_preferences", current_prefs)
            else:
                # Validate GPU ID
                total_gpus = self.get("gpu.total_gpu_count", 4)
                if not (0 <= gpu_id < total_gpus):
                    self.logger.error(f"Invalid GPU ID {gpu_id} for {total_gpus} total GPUs")
                    return False
                
                # Set preference
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
            agents = ["code_parser", "vector_index", "data_loader", 
                     "lineage_analyzer", "logic_analyzer", "documentation", "db2_comparator"]
            
            # Get configuration
            gpu_count = self.get("gpu.total_gpu_count", 4)
            exclude_gpu_0 = self.get("gpu.exclude_gpu_0", True)
            
            # Calculate assignment
            assignments = {}
            start_gpu = 1 if exclude_gpu_0 else 0
            available_gpus = gpu_count - start_gpu
            
            for i, agent in enumerate(agents):
                gpu_id = start_gpu + (i % available_gpus)
                assignments[agent] = gpu_id
                
                # Update configuration
                self.set_agent_preferred_gpu(agent, gpu_id)
            
            self.logger.info(f"Optimized GPU assignments: {assignments}")
            return assignments
            
        except Exception as e:
            self.logger.error(f"GPU assignment optimization failed: {e}")
            return {}
    
    def export_config(self, export_path: Optional[str] = None, format: str = "yaml") -> str:
        """Export configuration to string or file"""
        try:
            if format.lower() == "yaml":
                config_str = yaml.dump(self._config, default_flow_style=False, indent=2, sort_keys=False)
            elif format.lower() == "json":
                config_str = json.dumps(self._config, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            if export_path:
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(config_str)
                self.logger.info(f"Configuration exported to {export_path}")
            
            return config_str
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return ""
    
    def import_config(self, import_path: str, merge: bool = True) -> bool:
        """Import configuration from file"""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                self.logger.error(f"Import file does not exist: {import_path}")
                return False
            
            # Load import data
            if import_file.suffix.lower() in ['.yaml', '.yml']:
                with open(import_file, 'r', encoding='utf-8') as f:
                    import_data = yaml.safe_load(f)
            elif import_file.suffix.lower() == '.json':
                with open(import_file, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            else:
                self.logger.error(f"Unsupported import file format: {import_file.suffix}")
                return False
            
            if merge:
                # Merge with existing configuration
                backup_config = self._config.copy()
                try:
                    self._merge_config(self._config, import_data)
                    
                    # Validate merged configuration
                    issues = self.validate_config()
                    critical_errors = [issue for issue in issues if "ERROR" in issue]
                    
                    if critical_errors:
                        # Restore backup on critical errors
                        self._config = backup_config
                        self.logger.error(f"Import failed due to validation errors: {critical_errors}")
                        return False
                    
                except Exception as e:
                    self._config = backup_config
                    self.logger.error(f"Import merge failed: {e}")
                    return False
            else:
                # Replace entire configuration
                self._config = import_data
                
                # Validate imported configuration
                issues = self.validate_config()
                critical_errors = [issue for issue in issues if "ERROR" in issue]
                
                if critical_errors:
                    # Reload default config on critical errors
                    self._config = self._create_default_config()
                    self.logger.error(f"Import failed due to validation errors: {critical_errors}")
                    return False
            
            # Update metadata
            if "metadata" not in self._config:
                self._config["metadata"] = {}
            self._config["metadata"]["last_modified"] = datetime.now().isoformat()
            self._config["metadata"]["imported_from"] = str(import_path)
            
            self.logger.info(f"Configuration imported successfully from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        with self._lock:
            self._config = self._create_default_config()
            self._runtime_modifications.clear()
            self._validation_cache.clear()
            self.logger.info("Configuration reset to defaults")
    
    def backup_config(self) -> str:
        """Create backup of current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self.config_file.stem}_backup_{timestamp}{self.config_file.suffix}"
            backup_path = self.config_file.parent / backup_name
            
            # Create backup of in-memory config
            backup_config = self._config.copy()
            backup_config["metadata"]["backup_timestamp"] = timestamp
            backup_config["metadata"]["original_file"] = str(self.config_file)
            
            # Save backup
            if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    yaml.dump(backup_config, f, default_flow_style=False, indent=2)
            else:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_config, f, indent=2)
            
            self.logger.info(f"Configuration backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            return ""
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore configuration from backup"""
        try:
            return self.import_config(backup_path, merge=False)
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def apply_runtime_optimization(self, optimization_data: Dict[str, Any]):
        """Apply runtime optimizations without saving to file"""
        try:
            with self._lock:
                for key_path, value in optimization_data.items():
                    self._runtime_modifications[key_path] = value
                    
                self.logger.info(f"Applied {len(optimization_data)} runtime optimizations")
                
        except Exception as e:
            self.logger.error(f"Runtime optimization failed: {e}")
    
    def clear_runtime_modifications(self):
        """Clear all runtime modifications"""
        with self._lock:
            self._runtime_modifications.clear()
            self.logger.info("Runtime modifications cleared")
    
    def get_runtime_modifications(self) -> Dict[str, Any]:
        """Get current runtime modifications"""
        with self._lock:
            return self._runtime_modifications.copy()
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get comprehensive configuration information"""
        with self._lock:
            return {
                "config_file": str(self.config_file),
                "file_exists": self.config_file.exists(),
                "last_loaded": self._last_loaded,
                "last_modified": self._config.get("metadata", {}).get("last_modified"),
                "version": self._config.get("metadata", {}).get("version"),
                "validation_issues": self.validate_config(),
                "runtime_modifications": len(self._runtime_modifications),
                "sections": list(self._config.keys()),
                "total_settings": self._count_settings(self._config)
            }
    
    def _count_settings(self, config_dict: Dict[str, Any]) -> int:
        """Recursively count configuration settings"""
        count = 0
        for value in config_dict.values():
            if isinstance(value, dict):
                count += self._count_settings(value)
            else:
                count += 1
        return count
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has been modified"""
        if not self._auto_reload or not self.config_file.exists():
            return False
        
        try:
            file_mtime = self.config_file.stat().st_mtime
            if file_mtime > self._last_loaded:
                self.logger.info("Configuration file changed, reloading...")
                return self.load_config()
        except Exception as e:
            self.logger.error(f"Error checking file modification time: {e}")
        
        return False
    
    def enable_auto_reload(self, enabled: bool = True):
        """Enable or disable automatic configuration reloading"""
        self._auto_reload = enabled
        self.logger.info(f"Auto-reload {'enabled' if enabled else 'disabled'}")
    
    def get_environment_config(self, environment: str = None) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        if environment is None:
            environment = self.get("metadata.environment", "development")
        
        env_config = self._config.copy()
        
        # Apply environment-specific overrides
        env_overrides = {
            "production": {
                "logging.level": "WARNING",
                "performance.enable_gpu_monitoring": True,
                "optimization.enable_auto_optimization": True,
                "security.enable_auth": True
            },
            "development": {
                "logging.level": "DEBUG",
                "performance.enable_gpu_monitoring": True,
                "optimization.enable_auto_optimization": False
            },
            "testing": {
                "logging.level": "INFO",
                "performance.enable_gpu_monitoring": False,
                "optimization.enable_auto_optimization": False,
                "system.cache_ttl": 60
            }
        }
        
        if environment in env_overrides:
            for key_path, value in env_overrides[environment].items():
                self._set_nested_value(env_config, key_path, value)
        
        return env_config
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key_path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def create_profile(self, profile_name: str, base_profile: str = "default") -> bool:
        """Create configuration profile"""
        try:
            profiles_dir = self.config_file.parent / "profiles"
            profiles_dir.mkdir(exist_ok=True)
            
            profile_file = profiles_dir / f"{profile_name}{self.config_file.suffix}"
            
            if base_profile == "default":
                profile_config = self._create_default_config()
            else:
                base_file = profiles_dir / f"{base_profile}{self.config_file.suffix}"
                if not base_file.exists():
                    self.logger.error(f"Base profile {base_profile} does not exist")
                    return False
                profile_config = self.import_config(str(base_file), merge=False)
            
            # Save profile
            profile_config["metadata"]["profile_name"] = profile_name
            profile_config["metadata"]["base_profile"] = base_profile
            profile_config["metadata"]["created"] = datetime.now().isoformat()
            
            if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(profile_file, 'w', encoding='utf-8') as f:
                    yaml.dump(profile_config, f, default_flow_style=False, indent=2)
            else:
                with open(profile_file, 'w', encoding='utf-8') as f:
                    json.dump(profile_config, f, indent=2)
            
            self.logger.info(f"Profile {profile_name} created at {profile_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create profile {profile_name}: {e}")
            return False
    
    def load_profile(self, profile_name: str) -> bool:
        """Load configuration profile"""
        try:
            profiles_dir = self.config_file.parent / "profiles"
            profile_file = profiles_dir / f"{profile_name}{self.config_file.suffix}"
            
            if not profile_file.exists():
                self.logger.error(f"Profile {profile_name} does not exist")
                return False
            
            return self.import_config(str(profile_file), merge=False)
            
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_name}: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles"""
        try:
            profiles_dir = self.config_file.parent / "profiles"
            if not profiles_dir.exists():
                return []
            
            profiles = []
            for file in profiles_dir.glob(f"*{self.config_file.suffix}"):
                profiles.append(file.stem)
            
            return sorted(profiles)
            
        except Exception as e:
            self.logger.error(f"Failed to list profiles: {e}")
            return []


class DynamicConfigManager(ConfigManager):
    """Dynamic Configuration Manager with enhanced runtime optimization"""
    
    def __init__(self, config_file: str = "opulence_config.yaml"):
        super().__init__(config_file)
        self.optimization_history = []
        self.performance_metrics = {}
        self._optimization_lock = Lock()
    
    def create_runtime_config(self) -> Dict[str, Any]:
        """Create runtime configuration with current values including modifications"""
        with self._lock:
            # Start with base config
            runtime_config = self._config.copy()
            
            # Apply runtime modifications
            for key_path, value in self._runtime_modifications.items():
                self._set_nested_value(runtime_config, key_path, value)
            
            # Return flattened config for coordinator compatibility
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
                "utilization_threshold": self.get("gpu.utilization_threshold")
            }
    
    def record_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Record performance metric for optimization"""
        with self._optimization_lock:
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = []
            
            self.performance_metrics[metric_name].append({
                "timestamp": time.time(),
                "value": value,
                "context": context or {}
            })
            
            # Keep only recent metrics (last 100 entries)
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]
    
    def get_performance_trends(self, metric_name: str = None) -> Dict[str, Any]:
        """Get performance trends for optimization"""
        with self._optimization_lock:
            if metric_name:
                if metric_name in self.performance_metrics:
                    metrics = self.performance_metrics[metric_name]
                    values = [m["value"] for m in metrics]
                    
                    return {
                        "metric": metric_name,
                        "count": len(values),
                        "average": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "trend": "improving" if len(values) > 1 and values[-1] < values[0] else "stable",
                        "recent_values": values[-10:]  # Last 10 values
                    }
                else:
                    return {"metric": metric_name, "error": "Metric not found"}
            else:
                # Return all metrics summary
                summary = {}
                for name in self.performance_metrics:
                    summary[name] = self.get_performance_trends(name)
                return summary
    
    def auto_optimize_config(self) -> Dict[str, Any]:
        """Automatically optimize configuration based on performance metrics"""
        try:
            with self._optimization_lock:
                optimizations = {}
                
                # GPU memory optimization
                gpu_memory_metrics = self.performance_metrics.get("gpu_memory_usage", [])
                if gpu_memory_metrics:
                    recent_usage = [m["value"] for m in gpu_memory_metrics[-10:]]
                    avg_usage = sum(recent_usage) / len(recent_usage)
                    
                    current_threshold = self.get("gpu.memory_threshold", 0.80)
                    
                    if avg_usage > 0.90 and current_threshold > 0.70:
                        # Reduce memory threshold if usage is high
                        new_threshold = max(0.70, current_threshold - 0.05)
                        optimizations["gpu.memory_threshold"] = new_threshold
                    elif avg_usage < 0.60 and current_threshold < 0.85:
                        # Increase memory threshold if usage is low
                        new_threshold = min(0.85, current_threshold + 0.05)
                        optimizations["gpu.memory_threshold"] = new_threshold
                
                # Batch size optimization
                processing_time_metrics = self.performance_metrics.get("processing_time", [])
                if processing_time_metrics:
                    recent_times = [m["value"] for m in processing_time_metrics[-10:]]
                    avg_time = sum(recent_times) / len(recent_times)
                    
                    current_batch_size = self.get("system.batch_size", 32)
                    
                    if avg_time > 300 and current_batch_size > 16:  # 5 minutes
                        # Reduce batch size if processing is slow
                        new_batch_size = max(16, current_batch_size - 8)
                        optimizations["system.batch_size"] = new_batch_size
                    elif avg_time < 60 and current_batch_size < 64:  # 1 minute
                        # Increase batch size if processing is fast
                        new_batch_size = min(64, current_batch_size + 8)
                        optimizations["system.batch_size"] = new_batch_size
                
                # Cache TTL optimization
                cache_hit_metrics = self.performance_metrics.get("cache_hit_rate", [])
                if cache_hit_metrics:
                    recent_hits = [m["value"] for m in cache_hit_metrics[-10:]]
                    avg_hit_rate = sum(recent_hits) / len(recent_hits)
                    
                    current_ttl = self.get("system.cache_ttl", 3600)
                    
                    if avg_hit_rate < 0.3 and current_ttl > 1800:
                        # Reduce TTL if hit rate is low
                        new_ttl = max(1800, current_ttl - 600)
                        optimizations["system.cache_ttl"] = new_ttl
                    elif avg_hit_rate > 0.8 and current_ttl < 7200:
                        # Increase TTL if hit rate is high
                        new_ttl = min(7200, current_ttl + 600)
                        optimizations["system.cache_ttl"] = new_ttl
                
                # Apply optimizations as runtime modifications
                if optimizations:
                    self.apply_runtime_optimization(optimizations)
                    
                    # Record optimization
                    self.optimization_history.append({
                        "timestamp": time.time(),
                        "optimizations": optimizations,
                        "performance_snapshot": self.get_performance_trends()
                    })
                    
                    # Keep only recent optimization history
                    if len(self.optimization_history) > 50:
                        self.optimization_history = self.optimization_history[-50:]
                
                return {
                    "status": "success",
                    "optimizations_applied": len(optimizations),
                    "optimizations": optimizations,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Auto-optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        with self._optimization_lock:
            return self.optimization_history.copy()


# Global configuration manager instances
_config_manager = None
_dynamic_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_dynamic_config_manager() -> DynamicConfigManager:
    """Get global dynamic configuration manager instance"""
    global _dynamic_config_manager
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager()
    return _dynamic_config_manager

# Convenience functions for backward compatibility
def get_gpu_config() -> GPUConfig:
    """Get GPU configuration"""
    return get_config_manager().get_gpu_config()

def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return get_config_manager().get_system_config()

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return get_config_manager().get_performance_config()

def update_config(key_path: str, value: Any, persistent: bool = True) -> bool:
    """Update configuration value"""
    return get_config_manager().set(key_path, value, persistent)

def validate_current_config() -> List[str]:
    """Validate current configuration"""
    return get_config_manager().validate_config()

# Health monitoring utilities
class HealthMonitor:
    """Health monitoring for system components"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager or get_config_manager()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent,
                "status": "healthy" if memory.percent < 85 else "warning" if memory.percent < 95 else "critical"
            }
        except ImportError:
            # Fallback for systems without psutil
            return {
                "total_gb": 16.0,
                "available_gb": 8.0,
                "used_gb": 8.0,
                "percent_used": 50.0,
                "status": "unknown"
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            memory_status = self.get_memory_usage()
            config_issues = self.config_manager.validate_config()
            
            # Determine overall health
            if memory_status["status"] == "critical" or any("ERROR" in issue for issue in config_issues):
                overall_status = "critical"
            elif memory_status["status"] == "warning" or config_issues:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            return {
                "status": overall_status,
                "memory": memory_status,
                "configuration": {
                    "status": "healthy" if not config_issues else "warning",
                    "issues": config_issues
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Simple cache manager for configuration caching
class SimpleCacheManager:
    """Simple cache manager for configuration values"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache = {}
        self.timestamps = {}
        self._lock = Lock()
        
    def get(self, key: str) -> Any:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                return None
                
            # Check if expired
            if time.time() - self.timestamps.get(key, 0) > self.default_ttl:
                self.delete(key)
                return None
                
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        with self._lock:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "total_items": len(self.cache),
                "memory_usage_estimate": sum(len(str(v)) for v in self.cache.values()),
                "default_ttl": self.default_ttl
            }

# Export all classes and functions
__all__ = [
    'ConfigManager', 'DynamicConfigManager',
    'GPUConfig', 'SystemConfig', 'PerformanceConfig', 'OptimizationConfig',
    'DB2Config', 'SecurityConfig', 'LoggingConfig',
    'HealthMonitor', 'SimpleCacheManager',
    'get_config_manager', 'get_dynamic_config_manager',
    'get_gpu_config', 'get_system_config', 'get_performance_config',
    'update_config', 'validate_current_config'
]