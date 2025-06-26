# Fix 1: Update gpu_force_fix.py with better memory detection and cleanup

import os
import torch
import logging
import subprocess
import time
import gc

logger = logging.getLogger(__name__)

class GPUForcer:
    """Enhanced GPU forcing mechanism with memory management"""
    
    @staticmethod
    def check_gpu_memory(gpu_id: int) -> dict:
        """Check actual GPU memory with detailed process information"""
        try:
            # Use nvidia-smi to get detailed memory info
            result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            line = result.stdout.strip()
            if line:
                total, used, free = map(int, line.split(','))
                
                # Get process information
                process_result = subprocess.run([
                    'nvidia-smi', 
                    f'--id={gpu_id}',
                    '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                processes = []
                if process_result.stdout.strip():
                    for proc_line in process_result.stdout.strip().split('\n'):
                        if proc_line.strip():
                            try:
                                pid, mem = proc_line.split(',')
                                processes.append({
                                    'pid': int(pid.strip()),
                                    'memory_mb': int(mem.strip())
                                })
                            except:
                                pass
                
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'free_gb': free / 1024,
                    'total_gb': total / 1024,
                    'utilization_percent': (used / total) * 100,
                    'processes': processes,
                    'is_fragmented': used > 0 and free < 1000,  # Less than 1GB free but memory used
                    'is_available': free > 2000  # At least 2GB free
                }
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} memory: {e}")
        
        return {
            'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
            'free_gb': 0, 'total_gb': 0, 'utilization_percent': 100,
            'processes': [], 'is_fragmented': True, 'is_available': False
        }
    
    @staticmethod
    def find_best_gpu_with_memory(min_free_gb: float = 8.0, exclude_gpu_0: bool = True) -> Optional[int]:
        """Find GPU with sufficient free memory, optionally excluding GPU 0"""
        best_gpu = None
        best_free_memory = 0
        gpu_info = {}
        
        # Check all GPUs but start from GPU 1 if excluding GPU 0
        start_gpu = 1 if exclude_gpu_0 else 0
        
        for gpu_id in range(start_gpu, 4):  # Check GPUs 1-3 first, then 0 if needed
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            free_gb = memory_info['free_gb']
            gpu_info[gpu_id] = memory_info
            
            logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free, {memory_info['utilization_percent']:.1f}% used, "
                       f"processes: {len(memory_info['processes'])}, available: {memory_info['is_available']}")
            
            if memory_info['is_available'] and free_gb >= min_free_gb and free_gb > best_free_memory:
                best_gpu = gpu_id
                best_free_memory = free_gb
        
        # If no GPU found and we excluded GPU 0, try GPU 0 as last resort
        if best_gpu is None and exclude_gpu_0:
            logger.warning("No available GPU found excluding GPU 0, checking GPU 0 as last resort...")
            memory_info = GPUForcer.check_gpu_memory(0)
            gpu_info[0] = memory_info
            
            if memory_info['free_gb'] >= min_free_gb:
                logger.warning(f"Using GPU 0 as last resort with {memory_info['free_gb']:.1f}GB free")
                best_gpu = 0
                best_free_memory = memory_info['free_gb']
        
        if best_gpu is not None:
            logger.info(f"Selected GPU {best_gpu} with {best_free_memory:.1f}GB free")
        else:
            logger.error(f"No GPU found with at least {min_free_gb}GB free")
            logger.error("GPU Status Summary:")
            for gpu_id, info in gpu_info.items():
                logger.error(f"  GPU {gpu_id}: {info['free_gb']:.1f}GB free, "
                           f"{len(info['processes'])} processes, fragmented: {info['is_fragmented']}")
            
            # Try with lower requirement as emergency fallback
            for gpu_id, info in gpu_info.items():
                if info['free_gb'] >= 1.0:  # At least 1GB
                    logger.warning(f"EMERGENCY: Using GPU {gpu_id} with only {info['free_gb']:.1f}GB free")
                    return gpu_id
        
        return best_gpu
    
    @staticmethod
    def cleanup_gpu_memory(gpu_id: int, force: bool = False) -> bool:
        """Clean up GPU memory more aggressively"""
        try:
            logger.info(f"Cleaning up GPU {gpu_id} memory (force: {force})")
            
            # Set environment to target specific GPU
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            try:
                if torch.cuda.is_available():
                    # Force cleanup
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    if force:
                        # More aggressive cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Try to reset memory stats
                        try:
                            torch.cuda.reset_peak_memory_stats(0)
                            torch.cuda.reset_accumulated_memory_stats(0)
                        except:
                            pass
                
                # Wait for cleanup to take effect
                time.sleep(2)
                
                # Verify cleanup worked
                memory_info = GPUForcer.check_gpu_memory(gpu_id)
                logger.info(f"After cleanup - GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free")
                
                return memory_info['free_gb'] > 1.0  # At least 1GB should be free
                
            finally:
                # Restore environment
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    
        except Exception as e:
            logger.error(f"Error cleaning up GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def force_gpu_environment(gpu_id: int, cleanup_first: bool = True):
        """Aggressively force GPU environment with optional cleanup"""
        
        if cleanup_first:
            # Try to clean up the target GPU first
            GPUForcer.cleanup_gpu_memory(gpu_id, force=True)
        
        # Set CUDA environment to only show target GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # Set additional environment variables for memory management
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        
        # Set PyTorch device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 0 now maps to our target GPU
            logger.info(f"Set torch.cuda.set_device(0) -> actual GPU {gpu_id}")
    
    @staticmethod
    def create_vllm_engine_args(model_name: str, max_tokens: int = 4096, conservative_memory: bool = True):
        """Create VLLM engine args with conservative memory usage"""
        from vllm import AsyncEngineArgs
        
        # Use more conservative settings to avoid OOM
        memory_utilization = 0.50 if conservative_memory else 0.65  # Reduced from 0.65
        
        return AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_tokens,
            gpu_memory_utilization=memory_utilization,  # More conservative
            device="cuda:0",  # This is now our target GPU
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs to save memory
            disable_log_stats=True,
            quantization=None,
            load_format="auto",
            dtype="auto",
            seed=42,
            # Additional memory-saving options
            swap_space=2,  # 2GB swap space
            max_num_seqs=32,  # Reduced batch size
        )

def test_gpu_forcing():
    """Test the enhanced GPU forcing mechanism"""
    print("Testing Enhanced GPU Forcing Mechanism")
    print("=" * 50)
    
    # Check all GPUs first
    print("Checking all GPUs...")
    for gpu_id in range(4):
        memory_info = GPUForcer.check_gpu_memory(gpu_id)
        print(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free, "
              f"{memory_info['utilization_percent']:.1f}% used, "
              f"{len(memory_info['processes'])} processes")
    
    # Find best GPU (excluding GPU 0 by default)
    best_gpu = GPUForcer.find_best_gpu_with_memory(2.0, exclude_gpu_0=True)
    
    if best_gpu is None:
        print("❌ No suitable GPU found")
        return False
    
    print(f"✅ Found suitable GPU: {best_gpu}")
    
    # Force GPU environment
    print(f"🔧 Forcing GPU environment to use GPU {best_gpu}")
    GPUForcer.force_gpu_environment(best_gpu, cleanup_first=True)
    
    # Verify CUDA environment
    print(f"🔍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        print(f"🔍 PyTorch sees {device_count} device(s), current device: {current_device}")
        
        # Check memory on current device
        memory_info = torch.cuda.mem_get_info()
        free_gb = memory_info[0] / (1024**3)
        total_gb = memory_info[1] / (1024**3)
        print(f"🔍 Current device memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    
    return True

# Fix 2: Update coordinator to use smarter GPU selection

async def get_available_gpu_for_agent(self, agent_type: str, preferred_gpu: Optional[int] = None) -> Optional[int]:
    """Get best available GPU for agent with smart allocation and GPU 0 avoidance"""
    
    # Force refresh GPU status first
    self.gpu_manager.force_refresh()
    
    # Try to exclude GPU 0 unless specifically requested or no other option
    exclude_gpu_0 = (preferred_gpu != 0)  # Only use GPU 0 if specifically requested
    
    # Check if we already have an LLM engine that can be shared
    existing_engines = list(self.llm_engine_pool.keys())
    if existing_engines:
        for engine_key in existing_engines:
            gpu_id = int(engine_key.split('_')[1])
            
            # Skip GPU 0 if we're trying to avoid it
            if exclude_gpu_0 and gpu_id == 0:
                continue
                
            try:
                from gpu_force_fix import GPUForcer
                memory_info = GPUForcer.check_gpu_memory(gpu_id)
                free_gb = memory_info.get('free_gb', 0)
                
                # If GPU has enough free memory for sharing (reduced threshold)
                if free_gb >= 0.5:  # Very low threshold for sharing
                    self.logger.info(f"Reusing existing LLM engine on GPU {gpu_id} for {agent_type}")
                    return gpu_id
            except Exception as e:
                self.logger.warning(f"Error checking GPU {gpu_id} for reuse: {e}")
                continue
    
    # Use enhanced GPU finding with GPU 0 avoidance
    try:
        from gpu_force_fix import GPUForcer
        
        # First try excluding GPU 0
        if exclude_gpu_0:
            best_gpu = GPUForcer.find_best_gpu_with_memory(min_free_gb=1.5, exclude_gpu_0=True)
            if best_gpu is not None:
                self.logger.info(f"Selected GPU {best_gpu} for {agent_type} (avoided GPU 0)")
                return best_gpu
        
        # If preferred GPU is specified, try it
        if preferred_gpu is not None:
            memory_info = GPUForcer.check_gpu_memory(preferred_gpu)
            if memory_info.get('is_available', False):
                self.logger.info(f"Using preferred GPU {preferred_gpu} for {agent_type}")
                return preferred_gpu
        
        # Last resort: try any GPU including GPU 0
        best_gpu = GPUForcer.find_best_gpu_with_memory(min_free_gb=1.0, exclude_gpu_0=False)
        if best_gpu is not None:
            self.logger.warning(f"Using GPU {best_gpu} for {agent_type} (last resort)")
            return best_gpu
        
    except Exception as e:
        self.logger.error(f"Error in enhanced GPU selection: {e}")
    
    # Fallback to GPU manager
    best_gpu = self.gpu_manager.get_available_gpu(
        preferred_gpu=preferred_gpu, 
        fallback=True
    )
    
    if best_gpu is not None:
        self.logger.info(f"GPU manager selected GPU {best_gpu} for {agent_type}")
        return best_gpu
    
    self.logger.warning("No GPUs currently available")
    return None

# Fix 3: Update LLM engine creation with better memory management

async def get_or_create_llm_engine(self, gpu_id: int, force_reload: bool = False) -> AsyncLLMEngine:
    """Get or create LLM engine with enhanced memory management"""
    async with self.engine_lock:
        engine_key = f"gpu_{gpu_id}"
        
        # If engine exists and not forcing reload, return it
        if engine_key in self.llm_engine_pool and not force_reload:
            self.logger.info(f"Reusing existing LLM engine on GPU {gpu_id}")
            return self.llm_engine_pool[engine_key]
        
        try:
            from gpu_force_fix import GPUForcer
            
            # CHECK GPU AVAILABILITY with enhanced checking
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            free_gb = memory_info['free_gb']
            is_available = memory_info['is_available']
            is_fragmented = memory_info['is_fragmented']
            
            self.logger.info(f"GPU {gpu_id} status: {free_gb:.1f}GB free, available: {is_available}, "
                           f"fragmented: {is_fragmented}, processes: {len(memory_info['processes'])}")
            
            if not is_available:
                # Try cleanup first
                self.logger.warning(f"GPU {gpu_id} not available, attempting cleanup...")
                cleanup_success = GPUForcer.cleanup_gpu_memory(gpu_id, force=True)
                
                if cleanup_success:
                    # Recheck after cleanup
                    memory_info = GPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info['free_gb']
                    is_available = memory_info['is_available']
                
                if not is_available:
                    raise RuntimeError(f"GPU {gpu_id} is not available even after cleanup "
                                     f"(free: {free_gb:.1f}GB, processes: {len(memory_info['processes'])})")
            
            if free_gb < 1.5:  # Reduced requirement
                raise RuntimeError(f"GPU {gpu_id} has insufficient memory: {free_gb:.1f}GB free (need at least 1.5GB)")
            
            self.logger.info(f"Creating LLM engine on GPU {gpu_id} with {free_gb:.1f}GB free memory")
            
            # FORCE GPU ENVIRONMENT with cleanup
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                GPUForcer.force_gpu_environment(gpu_id, cleanup_first=True)
                
                # Create engine args with conservative memory settings
                engine_args = GPUForcer.create_vllm_engine_args(
                    self.config.model_name, 
                    self.config.max_tokens,
                    conservative_memory=True  # Use conservative memory settings
                )
                
                self.logger.info(f"Creating VLLM engine with conservative memory settings...")
                
                # Create the engine
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                
                self.llm_engine_pool[engine_key] = engine
                
                # Mark GPU as occupied in our manager  
                self.gpu_manager.reserve_gpu_for_workload(
                    workload_type=f"llm_engine_{engine_key}",
                    preferred_gpu=gpu_id,
                    duration_estimate=3600,  # 1 hour estimated
                    allow_sharing=True  # Allow sharing this GPU
                )
                
                # Verify final memory usage
                final_memory = GPUForcer.check_gpu_memory(gpu_id)
                final_free_gb = final_memory['free_gb']
                used_gb = free_gb - final_free_gb
                
                self.logger.info(f"✅ LLM engine created on GPU {gpu_id}. Used {used_gb:.1f}GB, {final_free_gb:.1f}GB remaining")
                
            finally:
                # Restore original CUDA_VISIBLE_DEVICES if it existed
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
        except Exception as e:
            self.logger.error(f"Failed to create LLM engine for GPU {gpu_id}: {str(e)}")
            
            # Clean up on failure
            GPUForcer.cleanup_gpu_memory(gpu_id, force=True)
            
            # Remove from pool if it was partially created
            if engine_key in self.llm_engine_pool:
                del self.llm_engine_pool[engine_key]
            
            # Release GPU workload on failure
            try:
                self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
            except:
                pass
            
            raise
        
        return self.llm_engine_pool[engine_key]