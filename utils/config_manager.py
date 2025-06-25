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