import weakref
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, Optional

class BaseOpulenceAgent:
    """Base class for all Opulence agents with automatic resource management - FIXED"""
    
    def __init__(self, coordinator, agent_type: str, db_path: str = "opulence_data.db", gpu_id: int = 0):
        self.coordinator = coordinator
        self.agent_type = agent_type
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # ✅ FIXED: Engine management with sharing awareness
        self._engine = None
        self._current_task_id = None
        self._engine_loaded = False
        self._using_shared_engine = False
        self._engine_shared_count = 0  # Track how many times this agent used shared engines
        
        # Register for automatic cleanup
        if coordinator:
            weakref.finalize(self, self._cleanup_callback, coordinator, agent_type, gpu_id)
    
    @staticmethod
    def _cleanup_callback(coordinator, agent_type: str, gpu_id: int):
        """Static cleanup callback for weakref"""
        try:
            if coordinator and gpu_id:
                coordinator.release_engine_reference(gpu_id)
                logging.getLogger(__name__).info(f"🧹 Auto-cleaned {agent_type} engine reference on GPU {gpu_id}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Auto-cleanup failed for {agent_type}: {e}")

    @asynccontextmanager
    async def get_engine_context(self):
        """FIXED: Context manager that checks for existing engines first"""
        
        # ✅ CRITICAL FIX: Check if we already have an active engine
        if self._engine is not None and self._current_task_id is not None:
            self.logger.info(f"♻️ {self.agent_type} reusing EXISTING context engine")
            self._engine_shared_count += 1
            yield self._engine
            return
        
        # ✅ CRITICAL FIX: Check assigned GPU and see if engine already exists
        assigned_gpu = self.coordinator.agent_gpu_assignments.get(self.agent_type)
        if assigned_gpu is None:
            assigned_gpu = self.coordinator.selected_gpus[0]  # fallback
        
        # ✅ KEY FIX: Check if engine already exists on assigned GPU
        if self.coordinator.gpu_manager.has_llm_engine(assigned_gpu):
            # Reuse existing engine without creating new task
            existing_engine = self.coordinator.gpu_manager.gpu_engines[assigned_gpu]
            
            # Create lightweight task tracking (not full task creation)
            task_id = f"{self.agent_type}_shared_{id(self)}"
            
            self._current_task_id = task_id
            self._engine = existing_engine
            self.gpu_id = assigned_gpu
            self._engine_loaded = True
            self._using_shared_engine = True
            self._engine_shared_count += 1
            
            self.logger.info(f"♻️ {self.agent_type} SHARING existing engine on GPU {assigned_gpu}")
            
            try:
                yield existing_engine
            finally:
                # Lightweight cleanup - don't call finish_task_with_engine
                self._current_task_id = None
                self._engine = None
                self._engine_loaded = False
        else:
            # ✅ Create new engine only if none exists
            task_id, engine, gpu_id = await self.coordinator.start_task_with_engine(
                f"{self.agent_type}_task", 
                self.agent_type,
                assigned_gpu
            )
            
            self._current_task_id = task_id
            self._engine = engine
            self.gpu_id = gpu_id
            self._engine_loaded = True
            self._using_shared_engine = True
            
            self.logger.info(f"🆕 {self.agent_type} CREATED new engine on GPU {gpu_id}")
            
            try:
                yield engine
            finally:
                # Full cleanup for newly created engines
                self.coordinator.finish_task_with_engine(task_id)
                self._current_task_id = None
                self._engine = None
                self._engine_loaded = False

    async def get_engine(self):
        """DEPRECATED: Get engine - prefer context manager for automatic cleanup"""
        self.logger.warning(f"⚠️ {self.agent_type} using deprecated get_engine() - use get_engine_context() instead")
        
        if self._engine is None and self.coordinator:
            # Check for existing engine first
            assigned_gpu = self.coordinator.agent_gpu_assignments.get(self.agent_type)
            if assigned_gpu is None:
                assigned_gpu = self.coordinator.selected_gpus[0]
            
            if self.coordinator.gpu_manager.has_llm_engine(assigned_gpu):
                # Reuse existing engine
                self._engine = self.coordinator.gpu_manager.gpu_engines[assigned_gpu]
                self.gpu_id = assigned_gpu
                self._engine_loaded = True
                self._using_shared_engine = True
                self._current_task_id = f"{self.agent_type}_direct_{id(self)}"
                
                self.logger.info(f"♻️ {self.agent_type} acquired EXISTING engine on GPU {assigned_gpu}")
            else:
                # Create new engine
                task_id, engine, gpu_id = await self.coordinator.start_task_with_engine(
                    f"{self.agent_type}_persistent", 
                    self.agent_type,
                    assigned_gpu
                )
                
                self._current_task_id = task_id
                self._engine = engine
                self.gpu_id = gpu_id
                self._engine_loaded = True
                self._using_shared_engine = True
                
                self.logger.info(f"🆕 {self.agent_type} acquired NEW engine on GPU {gpu_id}")
        
        return self._engine

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = self.agent_type
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_shared_count'] = self._engine_shared_count
            if self._current_task_id:
                result['task_id'] = self._current_task_id
        return result

    def cleanup(self):
        """Manual cleanup method"""
        if self._current_task_id:
            try:
                # Only call finish_task_with_engine for real tasks, not shared ones
                if not self._current_task_id.startswith(f"{self.agent_type}_shared_"):
                    self.coordinator.finish_task_with_engine(self._current_task_id)
                    self.logger.info(f"✅ {self.agent_type} cleaned up task {self._current_task_id}")
                else:
                    self.logger.info(f"✅ {self.agent_type} cleaned up shared context {self._current_task_id}")
            except Exception as e:
                self.logger.error(f"❌ Cleanup failed for {self.agent_type}: {e}")
            finally:
                self._current_task_id = None
                self._engine = None
                self._engine_loaded = False

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_type": self.agent_type,
            "gpu_id": self.gpu_id,
            "engine_loaded": self._engine_loaded,
            "using_shared_engine": self._using_shared_engine,
            "engine_shared_count": self._engine_shared_count,
            "has_active_task": self._current_task_id is not None,
            "current_task_id": self._current_task_id
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"type={self.agent_type}, "
                f"gpu={self.gpu_id}, "
                f"engine_loaded={self._engine_loaded}, "
                f"shared_count={self._engine_shared_count})")