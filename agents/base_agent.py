import weakref
from contextlib import asynccontextmanager

class BaseOpulenceAgent:
    """Base class for all Opulence agents with automatic resource management"""
    
    def __init__(self, coordinator, agent_type: str, db_path: str = "opulence_data.db", gpu_id: int = 0):
        self.coordinator = coordinator
        self.agent_type = agent_type
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # Engine management
        self._engine = None
        self._current_task_id = None
        self._engine_loaded = False
        self._using_shared_engine = False
        
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
        """Context manager for engine usage with automatic cleanup"""
        task_id, engine, gpu_id = await self.coordinator.start_task_with_engine(
            f"{self.agent_type}_task", 
            self.agent_type
        )
        
        self._current_task_id = task_id
        self._engine = engine
        self.gpu_id = gpu_id
        self._engine_loaded = True
        self._using_shared_engine = True
        
        try:
            yield engine
        finally:
            # Always cleanup, even if exception occurs
            self.coordinator.finish_task_with_engine(task_id)
            self._current_task_id = None
            self._engine = None
            self._engine_loaded = False

    async def get_engine(self):
        """Get engine - prefer context manager for automatic cleanup"""
        if self._engine is None and self.coordinator:
            # Start a task and get engine
            task_id, engine, gpu_id = await self.coordinator.start_task_with_engine(
                f"{self.agent_type}_persistent", 
                self.agent_type
            )
            
            self._current_task_id = task_id
            self._engine = engine
            self.gpu_id = gpu_id
            self._engine_loaded = True
            self._using_shared_engine = True
            
            self.logger.info(f"✅ {self.agent_type} acquired engine on GPU {gpu_id}")
        
        return self._engine

    def cleanup(self):
        """Manual cleanup method"""
        if self._current_task_id:
            try:
                self.coordinator.finish_task_with_engine(self._current_task_id)
                self.logger.info(f"✅ {self.agent_type} cleaned up task {self._current_task_id}")
            except Exception as e:
                self.logger.error(f"❌ Cleanup failed for {self.agent_type}: {e}")
            finally:
                self._current_task_id = None
                self._engine = None
                self._engine_loaded = False