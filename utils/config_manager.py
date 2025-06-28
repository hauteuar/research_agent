# config.py - Simple Configuration Module
"""
Simple configuration management for the Opulence system
Provides basic configuration with sensible defaults
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU configuration settings"""
    total_gpu_count: int = 4
    memory_threshold: float = 0.80
    utilization_threshold: float = 75.0
    exclude_gpu_0: bool = True
    conservative_memory: bool = True
    max_concurrent_requests: int = 2

@dataclass
class SystemConfig:
    """System configuration settings"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 4096
    temperature: float = 0.1
    batch_size: int = 32
    max_processing_time: int = 900  # 15 minutes
    cache_ttl: int = 3600  # 1 hour
    db_path: str = "opulence_data.db"
    log_level: str = "INFO"

@dataclass
class AgentConfig:
    """Agent configuration settings"""
    enable_llm: bool = True
    conservative_mode: bool = True
    timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    preferred_gpu_mapping: Dict[str, Optional[int]] = None

    def __post_init__(self):
        if self.preferred_gpu_mapping is None:
            self.preferred_gpu_mapping = {
                "code_parser": None,
                "vector_index": None,
                "data_loader": None,
                "lineage_analyzer": None,
                "logic_analyzer": None,
                "documentation": None,
                "db2_comparator": None
            }

class ConfigManager:
    """Simple configuration manager"""
    
    def __init__(self, config_file: str = "opulence_config.json"):
        self.config_file = Path(config_file)
        self.gpu_config = GPUConfig()
        self.system_config = SystemConfig()
        self.agent_config = AgentConfig()
        
        # Load configuration if file exists
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update GPU config
                if 'gpu' in config_data:
                    gpu_data = config_data['gpu']
                    for key, value in gpu_data.items():
                        if hasattr(self.gpu_config, key):
                            setattr(self.gpu_config, key, value)
                
                # Update system config
                if 'system' in config_data:
                    system_data = config_data['system']
                    for key, value in system_data.items():
                        if hasattr(self.system_config, key):
                            setattr(self.system_config, key, value)
                
                # Update agent config
                if 'agents' in config_data:
                    agent_data = config_data['agents']
                    for key, value in agent_data.items():
                        if hasattr(self.agent_config, key):
                            setattr(self.agent_config, key, value)
                
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info("No configuration file found, using defaults")
                self.save_config()  # Create default config file
                return True
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                'gpu': asdict(self.gpu_config),
                'system': asdict(self.system_config),
                'agents': asdict(self.agent_config)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_gpu_config(self) -> GPUConfig:
        """Get GPU configuration"""
        return self.gpu_config
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.system_config
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        return self.agent_config
    
    def update_gpu_config(self, **kwargs) -> bool:
        """Update GPU configuration"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.gpu_config, key):
                    setattr(self.gpu_config, key, value)
                    logger.info(f"Updated GPU config: {key} = {value}")
                else:
                    logger.warning(f"Unknown GPU config parameter: {key}")
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to update GPU configuration: {e}")
            return False
    
    def update_system_config(self, **kwargs) -> bool:
        """Update system configuration"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.system_config, key):
                    setattr(self.system_config, key, value)
                    logger.info(f"Updated system config: {key} = {value}")
                else:
                    logger.warning(f"Unknown system config parameter: {key}")
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to update system configuration: {e}")
            return False
    
    def set_agent_gpu_preference(self, agent_type: str, gpu_id: Optional[int]) -> bool:
        """Set GPU preference for specific agent"""
        try:
            if agent_type in self.agent_config.preferred_gpu_mapping:
                self.agent_config.preferred_gpu_mapping[agent_type] = gpu_id
                logger.info(f"Set GPU preference for {agent_type}: {gpu_id}")
                return self.save_config()
            else:
                logger.warning(f"Unknown agent type: {agent_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set agent GPU preference: {e}")
            return False
    
    def get_agent_gpu_preference(self, agent_type: str) -> Optional[int]:
        """Get GPU preference for specific agent"""
        return self.agent_config.preferred_gpu_mapping.get(agent_type)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'gpu': asdict(self.gpu_config),
            'system': asdict(self.system_config),
            'agents': asdict(self.agent_config)
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate GPU config
        if self.gpu_config.total_gpu_count < 1:
            issues.append("GPU count must be at least 1")
        
        if not (0.1 <= self.gpu_config.memory_threshold <= 1.0):
            issues.append("Memory threshold must be between 0.1 and 1.0")
        
        if not (0.1 <= self.gpu_config.utilization_threshold <= 100.0):
            issues.append("Utilization threshold must be between 0.1 and 100.0")
        
        # Validate system config
        if self.system_config.max_tokens < 1:
            issues.append("Max tokens must be at least 1")
        
        if not (0.0 <= self.system_config.temperature <= 2.0):
            issues.append("Temperature must be between 0.0 and 2.0")
        
        if self.system_config.batch_size < 1:
            issues.append("Batch size must be at least 1")
        
        # Validate agent config
        if self.agent_config.timeout < 1:
            issues.append("Agent timeout must be at least 1 second")
        
        if self.agent_config.retry_attempts < 0:
            issues.append("Retry attempts must be non-negative")
        
        return issues
    
    def reset_to_defaults(self) -> bool:
        """Reset all configuration to defaults"""
        try:
            self.gpu_config = GPUConfig()
            self.system_config = SystemConfig()
            self.agent_config = AgentConfig()
            
            logger.info("Configuration reset to defaults")
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def backup_config(self) -> str:
        """Create backup of current configuration"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.config_file.stem}_backup_{timestamp}.json"
            
            config_data = self.get_all_config()
            with open(backup_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration backed up to {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return ""

# =====================================
# GLOBAL CONFIG INSTANCE
# =====================================

# Global configuration manager
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_gpu_config() -> GPUConfig:
    """Get GPU configuration"""
    return get_config_manager().get_gpu_config()

def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return get_config_manager().get_system_config()

def get_agent_config() -> AgentConfig:
    """Get agent configuration"""
    return get_config_manager().get_agent_config()

# =====================================
# CONVENIENCE FUNCTIONS
# =====================================

def update_gpu_settings(**kwargs) -> bool:
    """Update GPU settings"""
    return get_config_manager().update_gpu_config(**kwargs)

def update_system_settings(**kwargs) -> bool:
    """Update system settings"""
    return get_config_manager().update_system_config(**kwargs)

def set_agent_gpu(agent_type: str, gpu_id: Optional[int]) -> bool:
    """Set GPU preference for agent"""
    return get_config_manager().set_agent_gpu_preference(agent_type, gpu_id)

def get_agent_gpu(agent_type: str) -> Optional[int]:
    """Get GPU preference for agent"""
    return get_config_manager().get_agent_gpu_preference(agent_type)

def validate_current_config() -> List[str]:
    """Validate current configuration"""
    return get_config_manager().validate_config()

def save_current_config() -> bool:
    """Save current configuration"""
    return get_config_manager().save_config()

def load_config_from_file(config_file: str = None) -> bool:
    """Load configuration from specific file"""
    if config_file:
        global _config_manager
        _config_manager = ConfigManager(config_file)
        return _config_manager.load_config()
    else:
        return get_config_manager().load_config()

def create_default_config(config_file: str = "opulence_config.json") -> bool:
    """Create default configuration file"""
    try:
        config_manager = ConfigManager(config_file)
        return config_manager.save_config()
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        return False

# =====================================
# CONFIGURATION TEMPLATES
# =====================================

def get_high_performance_config() -> Dict[str, Any]:
    """Get high performance configuration template"""
    return {
        'gpu': {
            'total_gpu_count': 4,
            'memory_threshold': 0.90,
            'utilization_threshold': 85.0,
            'exclude_gpu_0': False,  # Use all GPUs for max performance
            'conservative_memory': False,
            'max_concurrent_requests': 4
        },
        'system': {
            'model_name': "codellama/CodeLlama-7b-Instruct-hf",
            'max_tokens': 8192,  # Larger context
            'temperature': 0.05,  # More deterministic
            'batch_size': 64,  # Larger batches
            'max_processing_time': 1800,  # 30 minutes
            'cache_ttl': 7200  # 2 hours
        },
        'agents': {
            'enable_llm': True,
            'conservative_mode': False,
            'timeout': 600,  # 10 minutes
            'retry_attempts': 5
        }
    }

def get_conservative_config() -> Dict[str, Any]:
    """Get conservative configuration template"""
    return {
        'gpu': {
            'total_gpu_count': 4,
            'memory_threshold': 0.70,
            'utilization_threshold': 60.0,
            'exclude_gpu_0': True,  # Avoid GPU 0
            'conservative_memory': True,
            'max_concurrent_requests': 1
        },
        'system': {
            'model_name': "codellama/CodeLlama-7b-Instruct-hf",
            'max_tokens': 2048,  # Smaller context
            'temperature': 0.1,
            'batch_size': 16,  # Smaller batches
            'max_processing_time': 600,  # 10 minutes
            'cache_ttl': 1800  # 30 minutes
        },
        'agents': {
            'enable_llm': True,
            'conservative_mode': True,
            'timeout': 180,  # 3 minutes
            'retry_attempts': 2
        }
    }
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
    

def apply_config_template(template_name: str) -> bool:
    """Apply a configuration template"""
    try:
        config_manager = get_config_manager()
        
        if template_name == "high_performance":
            template = get_high_performance_config()
        elif template_name == "conservative":
            template = get_conservative_config()
        else:
            logger.error(f"Unknown template: {template_name}")
            return False
        
        # Apply GPU config
        config_manager.update_gpu_config(**template['gpu'])
        
        # Apply system config
        config_manager.update_system_config(**template['system'])
        
        # Apply agent config
        for key, value in template['agents'].items():
            if hasattr(config_manager.agent_config, key):
                setattr(config_manager.agent_config, key, value)
        
        logger.info(f"Applied {template_name} configuration template")
        return config_manager.save_config()
        
    except Exception as e:
        logger.error(f"Failed to apply template {template_name}: {e}")
        return False

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

# =====================================
# TESTING FUNCTION
# =====================================

def test_config_manager():
    """Test the configuration manager"""
    print("🚀 Testing Configuration Manager")
    print("=" * 50)
    
    # Create config manager
    config_manager = ConfigManager("test_config.json")
    
    # Test validation
    issues = config_manager.validate_config()
    if issues:
        print(f"❌ Validation issues: {issues}")
    else:
        print("✅ Configuration validation passed")
    
    # Test updates
    success = config_manager.update_gpu_config(total_gpu_count=3, memory_threshold=0.75)
    print(f"✅ GPU config update: {'Success' if success else 'Failed'}")
    
    # Test agent GPU preferences
    success = config_manager.set_agent_gpu_preference("code_parser", 1)
    print(f"✅ Agent GPU preference: {'Success' if success else 'Failed'}")
    
    # Test templates
    success = apply_config_template("conservative")
    print(f"✅ Template application: {'Success' if success else 'Failed'}")
    
    # Test backup
    backup_file = config_manager.backup_config()
    print(f"✅ Config backup: {backup_file if backup_file else 'Failed'}")
    
    print("\n✅ Configuration manager test completed")

if __name__ == "__main__":
    # Run tests if script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_config_manager()
config_manager = ConfigManager()