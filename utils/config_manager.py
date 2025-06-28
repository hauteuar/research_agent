# utils/config_manager.py
"""
Configuration Manager for Opulence Deep Research Mainframe Agent
Handles YAML/JSON configuration files and environment variables
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

@dataclass
class SystemConfig:
    """System configuration settings"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 4096
    temperature: float = 0.1
    gpu_count: int = 3
    max_processing_time: int = 900
    batch_size: int = 32
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600

@dataclass  
class DB2Config:
    """DB2 database configuration"""
    database: str = "TESTDB"
    hostname: str = "localhost"
    port: str = "50000"
    username: str = "db2user"
    password: str = "password"
    connection_timeout: int = 30

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: str = "opulence.log"
    max_size_mb: int = 100
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool = False
    session_timeout: int = 3600
    allowed_file_types: List[str] = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = [
                ".cbl", ".cob", ".jcl", ".csv", ".ddl", 
                ".sql", ".dcl", ".copy", ".cpy", ".zip"
            ]

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_gpu_monitoring: bool = True
    health_check_interval: int = 60
    cleanup_interval: int = 300

class ConfigManager:
    """Manages system configuration with file-based and environment override support"""
    
    def __init__(self, config_file: str = "config/opulence_config.yaml"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default configurations
        self.system = SystemConfig()
        self.db2 = DB2Config()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Load configuration from file if exists
        if self.config_file.exists():
            self.load_config()
        else:
            # Create default config file
            self._create_default_config_file()
        
        # Override with environment variables
        self._load_environment_overrides()
    
    def _create_default_config_file(self):
        """Create default configuration file"""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            default_config = {
                "system": asdict(self.system),
                "db2": asdict(self.db2),
                "logging": asdict(self.logging),
                "security": asdict(self.security),
                "performance": asdict(self.performance)
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Created default configuration file: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default config file: {str(e)}")
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() == '.yaml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configuration objects
            if 'system' in config_data:
                self.system = SystemConfig(**config_data['system'])
            
            if 'db2' in config_data:
                self.db2 = DB2Config(**config_data['db2'])
            
            if 'logging' in config_data:
                self.logging = LoggingConfig(**config_data['logging'])
            
            if 'security' in config_data:
                self.security = SecurityConfig(**config_data['security'])
            
            if 'performance' in config_data:
                self.performance = PerformanceConfig(**config_data['performance'])
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_file}: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                "system": asdict(self.system),
                "db2": asdict(self.db2),
                "logging": asdict(self.logging),
                "security": asdict(self.security),
                "performance": asdict(self.performance)
            }
            
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                if self.config_file.suffix.lower() == '.yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_file}: {str(e)}")
            return False
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # System overrides
        if os.getenv('OPULENCE_MODEL_NAME'):
            self.system.model_name = os.getenv('OPULENCE_MODEL_NAME')
        
        if os.getenv('OPULENCE_GPU_COUNT'):
            try:
                self.system.gpu_count = int(os.getenv('OPULENCE_GPU_COUNT'))
            except ValueError:
                pass
        
        if os.getenv('OPULENCE_MAX_TOKENS'):
            try:
                self.system.max_tokens = int(os.getenv('OPULENCE_MAX_TOKENS'))
            except ValueError:
                pass
        
        if os.getenv('OPULENCE_TEMPERATURE'):
            try:
                self.system.temperature = float(os.getenv('OPULENCE_TEMPERATURE'))
            except ValueError:
                pass
        
        # DB2 overrides
        if os.getenv('DB2_DATABASE'):
            self.db2.database = os.getenv('DB2_DATABASE')
        
        if os.getenv('DB2_HOSTNAME'):
            self.db2.hostname = os.getenv('DB2_HOSTNAME')
        
        if os.getenv('DB2_PORT'):
            self.db2.port = os.getenv('DB2_PORT')
        
        if os.getenv('DB2_USERNAME'):
            self.db2.username = os.getenv('DB2_USERNAME')
        
        if os.getenv('DB2_PASSWORD'):
            self.db2.password = os.getenv('DB2_PASSWORD')
        
        # Logging overrides
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        
        if os.getenv('LOG_FILE'):
            self.logging.file = os.getenv('LOG_FILE')
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'system.gpu_count')"""
        try:
            keys = key_path.split('.')
            
            if keys[0] == 'system':
                obj = self.system
            elif keys[0] == 'db2':
                obj = self.db2
            elif keys[0] == 'logging':
                obj = self.logging
            elif keys[0] == 'security':
                obj = self.security
            elif keys[0] == 'performance':
                obj = self.performance
            else:
                return default
            
            # Navigate through remaining keys
            for key in keys[1:]:
                obj = getattr(obj, key)
            
            return obj
            
        except (AttributeError, IndexError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            
            if keys[0] == 'system':
                obj = self.system
            elif keys[0] == 'db2':
                obj = self.db2
            elif keys[0] == 'logging':
                obj = self.logging
            elif keys[0] == 'security':
                obj = self.security
            elif keys[0] == 'performance':
                obj = self.performance
            else:
                return False
            
            # Navigate to parent and set attribute
            for key in keys[1:-1]:
                obj = getattr(obj, key)
            
            setattr(obj, keys[-1], value)
            return True
            
        except (AttributeError, IndexError):
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section as dictionary"""
        if section == 'system':
            return asdict(self.system)
        elif section == 'db2':
            return asdict(self.db2)
        elif section == 'logging':
            return asdict(self.logging)
        elif section == 'security':
            return asdict(self.security)
        elif section == 'performance':
            return asdict(self.performance)
        else:
            return {}
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section with new values"""
        try:
            if section == 'system':
                for key, value in updates.items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)
            elif section == 'db2':
                for key, value in updates.items():
                    if hasattr(self.db2, key):
                        setattr(self.db2, key, value)
            elif section == 'logging':
                for key, value in updates.items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
            elif section == 'security':
                for key, value in updates.items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
            elif section == 'performance':
                for key, value in updates.items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)
            else:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update section {section}: {str(e)}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate system configuration
        if self.system.gpu_count < 1:
            issues.append("GPU count must be at least 1")
        
        if self.system.max_processing_time < 60:
            issues.append("Max processing time should be at least 60 seconds")
        
        if self.system.batch_size < 1:
            issues.append("Batch size must be at least 1")
        
        if not (0.0 <= self.system.temperature <= 2.0):
            issues.append("Temperature should be between 0.0 and 2.0")
        
        if self.system.max_tokens < 100:
            issues.append("Max tokens should be at least 100")
        
        # Validate DB2 configuration
        if not self.db2.database:
            issues.append("DB2 database name is required")
        
        if not self.db2.hostname:
            issues.append("DB2 hostname is required")
        
        if not self.db2.username:
            issues.append("DB2 username is required")
        
        try:
            port = int(self.db2.port)
            if not (1 <= port <= 65535):
                issues.append("DB2 port must be between 1 and 65535")
        except ValueError:
            issues.append("DB2 port must be a valid number")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level not in valid_log_levels:
            issues.append(f"Log level must be one of: {', '.join(valid_log_levels)}")
        
        if self.logging.max_size_mb < 1:
            issues.append("Log max size must be at least 1 MB")
        
        # Validate security configuration
        if self.security.session_timeout < 300:
            issues.append("Session timeout should be at least 300 seconds")
        
        # Validate performance configuration
        if self.performance.cache_size < 10:
            issues.append("Cache size should be at least 10")
        
        if self.performance.health_check_interval < 10:
            issues.append("Health check interval should be at least 10 seconds")
        
        return issues
    
    def export_config(self, export_path: str = None) -> str:
        """Export configuration to string or file"""
        config_data = {
            "system": asdict(self.system),
            "db2": asdict(self.db2),
            "logging": asdict(self.logging),
            "security": asdict(self.security),
            "performance": asdict(self.performance)
        }
        
        config_str = yaml.dump(config_data, default_flow_style=False, indent=2)
        
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
        self.system = SystemConfig()
        self.db2 = DB2Config()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.logger.info("Configuration reset to defaults")
    
    def get_db2_connection_string(self) -> str:
        """Get DB2 connection string"""
        return f"""
            DATABASE={self.db2.database};
            HOSTNAME={self.db2.hostname};
            PORT={self.db2.port};
            PROTOCOL=TCPIP;
            UID={self.db2.username};
            PWD={self.db2.password};
        """.strip()
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode"""
        return os.getenv('OPULENCE_ENV', 'development').lower() == 'production'
    
    def get_data_directory(self) -> Path:
        """Get data directory path"""
        data_dir = os.getenv('OPULENCE_DATA_DIR', 'data')
        return Path(data_dir)
    
    def get_log_directory(self) -> Path:
        """Get log directory path"""
        log_dir = os.getenv('OPULENCE_LOG_DIR', 'logs')
        return Path(log_dir)
    
    def get_cache_directory(self) -> Path:
        """Get cache directory path"""
        cache_dir = os.getenv('OPULENCE_CACHE_DIR', 'cache')
        return Path(cache_dir)


# Global configuration instance
config_manager = ConfigManager()