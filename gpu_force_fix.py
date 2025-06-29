import os
import torch
import logging
import subprocess
import time
import gc
import psutil
from typing import Optional, Dict, List
import threading
import signal

logger = logging.getLogger(__name__)

class EnhancedGPUForcer:
    """Enhanced GPU forcing with process management and memory optimization"""
    
    _gpu_locks = {}
    _process_registry = {}
    
    @classmethod
    def init_gpu_locks(cls, gpu_count: int = 4):
        """Initialize locks for each GPU"""
        for i in range(gpu_count):
            if i not in cls._gpu_locks:
                cls._gpu_locks[i] = threading.Lock()
    
    @staticmethod
    def check_system_resources() -> Dict:
        """Check overall system resource usage"""
        try:
            # Check CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            process_count = len(psutil.pids())
            
            # Check for zombie processes
            zombie_count = 0
            python_processes = 0
            cuda_processes = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                    if 'cuda' in proc.info['name'].lower():
                        cuda_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'process_count': process_count,
                'zombie_processes': zombie_count,
                'python_processes': python_processes,
                'cuda_processes': cuda_processes,
                'critical_load': process_count > 1000 or memory.percent > 90 or cpu_percent > 95
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {'critical_load': True, 'error': str(e)}
    
    @staticmethod
    def cleanup_zombie_processes():
        """Clean up zombie processes"""
        try:
            # Kill zombie Python processes related to CUDA/torch
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cmdline']):
                try:
                    if (proc.info['status'] == psutil.STATUS_ZOMBIE and 
                        'python' in proc.info['name'].lower()):
                        logger.warning(f"Killing zombie process {proc.info['pid']}")
                        proc.kill()
                        
                    # Also kill hanging CUDA processes
                    if (proc.info['name'] and 
                        any(cuda_term in proc.info['name'].lower() 
                            for cuda_term in ['cuda', 'nvidia', 'vllm']) and
                        proc.info.get('cmdline')):
                        # Check if process is consuming too much CPU without progress
                        if proc.cpu_percent() > 50:
                            logger.warning(f"Killing high-CPU CUDA process {proc.info['pid']}")
                            proc.terminate()
                            time.sleep(2)
                            if proc.is_running():
                                proc.kill()
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
                    
        except Exception as e:
            logger.error(f"Zombie cleanup failed: {e}")
    
    @staticmethod
    def check_gpu_memory_detailed(gpu_id: int) -> Dict:
        """Enhanced GPU memory checking with process details"""
        try:
            # Get basic memory info
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True, timeout=10)
            
            line = result.stdout.strip()
            if line:
                total, used, free, utilization, temperature = map(int, line.split(','))
                
                # Get detailed process information
                process_result = subprocess.run([
                    'nvidia-smi', f'--id={gpu_id}',
                    '--query-compute-apps=pid,used_memory,process_name',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=10)
                
                processes = []
                total_process_memory = 0
                
                if process_result.stdout.strip():
                    for proc_line in process_result.stdout.strip().split('\n'):
                        if proc_line.strip():
                            try:
                                parts = proc_line.split(',')
                                if len(parts) >= 2:
                                    pid = int(parts[0].strip())
                                    mem = int(parts[1].strip())
                                    name = parts[2].strip() if len(parts) > 2 else "unknown"
                                    
                                    processes.append({
                                        'pid': pid,
                                        'memory_mb': mem,
                                        'process_name': name
                                    })
                                    total_process_memory += mem
                            except (ValueError, IndexError):
                                continue
                
                # Calculate fragmentation
                memory_efficiency = (total_process_memory / used) if used > 0 else 1.0
                is_fragmented = memory_efficiency < 0.8 and used > 1000
                
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'free_gb': free / 1024,
                    'total_gb': total / 1024,
                    'utilization_percent': utilization,
                    'temperature': temperature,
                    'processes': processes,
                    'process_count': len(processes),
                    'total_process_memory': total_process_memory,
                    'memory_efficiency': memory_efficiency,
                    'is_fragmented': is_fragmented,
                    'is_available': free > 2000 and utilization < 80 and temperature < 85,
                    'is_overheated': temperature > 85,
                    'needs_cleanup': is_fragmented or len(processes) > 5
                }
        except subprocess.TimeoutExpired:
            logger.error(f"nvidia-smi timeout for GPU {gpu_id}")
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} memory: {e}")
        
        return {
            'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
            'free_gb': 0, 'total_gb': 0, 'utilization_percent': 100,
            'temperature': 100, 'processes': [], 'process_count': 0,
            'total_process_memory': 0, 'memory_efficiency': 0,
            'is_fragmented': True, 'is_available': False,
            'is_overheated': True, 'needs_cleanup': True
        }
    
    @staticmethod
    def aggressive_gpu_cleanup(gpu_id: int) -> bool:
        """Aggressive GPU cleanup including process termination"""
        logger.info(f"Starting aggressive cleanup for GPU {gpu_id}")
        
        try:
            # 1. Get current processes on GPU
            memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
            processes = memory_info.get('processes', [])
            
            # 2. Kill hanging processes
            killed_processes = []
            for proc_info in processes:
                try:
                    pid = proc_info['pid']
                    proc = psutil.Process(pid)
                    
                    # Check if process is hanging (high memory, low CPU for extended time)
                    cpu_percent = proc.cpu_percent()
                    if cpu_percent < 1.0 and proc_info['memory_mb'] > 500:
                        logger.warning(f"Killing hanging GPU process PID {pid}: {proc_info['process_name']}")
                        proc.terminate()
                        killed_processes.append(pid)
                        
                        # Wait and force kill if needed
                        time.sleep(2)
                        if proc.is_running():
                            proc.kill()
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                except Exception as e:
                    logger.warning(f"Error killing process {pid}: {e}")
            
            # 3. Clear Python/PyTorch memory
            if torch.cuda.is_available():
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(gpu_id)
                        torch.cuda.reset_accumulated_memory_stats(gpu_id)
                        
                        # Force garbage collection
                        gc.collect()
                        
                except Exception as e:
                    logger.warning(f"PyTorch cleanup failed for GPU {gpu_id}: {e}")
            
            # 4. System-level cleanup
            gc.collect()
            
            # 5. Wait and verify cleanup
            time.sleep(3)
            new_memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
            cleanup_success = new_memory_info['free_gb'] > memory_info['free_gb']
            
            logger.info(f"GPU {gpu_id} cleanup: {memory_info['free_gb']:.1f}GB -> {new_memory_info['free_gb']:.1f}GB free")
            logger.info(f"Killed processes: {killed_processes}")
            
            return cleanup_success
            
        except Exception as e:
            logger.error(f"Aggressive GPU cleanup failed for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def find_optimal_gpu(min_free_gb: float = 6.0, exclude_gpu_0: bool = True) -> Optional[int]:
        """Find optimal GPU with comprehensive checks"""
        
        # First check system resources
        system_status = EnhancedGPUForcer.check_system_resources()
        
        if system_status.get('critical_load', False):
            logger.warning("System under critical load, attempting cleanup...")
            EnhancedGPUForcer.cleanup_zombie_processes()
            time.sleep(2)
        
        gpu_candidates = []
        gpu_info = {}
        
        # Check all GPUs
        start_gpu = 1 if exclude_gpu_0 else 0
        for gpu_id in range(start_gpu, 4):
            memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
            gpu_info[gpu_id] = memory_info
            
            # Score GPU based on multiple factors
            score = 0
            free_gb = memory_info['free_gb']
            
            if memory_info['is_available'] and free_gb >= min_free_gb:
                score += free_gb * 10  # Free memory weight
                score -= memory_info['utilization_percent'] / 10  # Lower utilization better
                score -= memory_info['temperature'] / 100  # Lower temperature better
                score -= memory_info['process_count'] * 5  # Fewer processes better
                score += memory_info['memory_efficiency'] * 20  # Higher efficiency better
                
                # Penalty for fragmentation
                if memory_info['is_fragmented']:
                    score -= 20
                
                # Penalty for overheating
                if memory_info['is_overheated']:
                    score -= 50
                
                gpu_candidates.append((gpu_id, score, memory_info))
        
        # Sort by score (highest first)
        gpu_candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("GPU Scoring Results:")
        for gpu_id, score, info in gpu_candidates[:3]:
            logger.info(f"  GPU {gpu_id}: Score {score:.1f}, {info['free_gb']:.1f}GB free, "
                       f"{info['utilization_percent']}% util, {info['process_count']} procs")
        
        # Try top candidate
        if gpu_candidates:
            best_gpu = gpu_candidates[0][0]
            best_info = gpu_candidates[0][2]
            
            # If best GPU needs cleanup, try it
            if best_info.get('needs_cleanup', False):
                logger.info(f"Best GPU {best_gpu} needs cleanup, attempting...")
                cleanup_success = EnhancedGPUForcer.aggressive_gpu_cleanup(best_gpu)
                if cleanup_success:
                    # Recheck after cleanup
                    new_info = EnhancedGPUForcer.check_gpu_memory_detailed(best_gpu)
                    if new_info['free_gb'] >= min_free_gb:
                        logger.info(f"✅ GPU {best_gpu} ready after cleanup: {new_info['free_gb']:.1f}GB free")
                        return best_gpu
            else:
                logger.info(f"✅ Selected optimal GPU {best_gpu}: {best_info['free_gb']:.1f}GB free")
                return best_gpu
        
        # Last resort: try GPU 0 if excluded
        if exclude_gpu_0:
            logger.warning("No suitable GPU found, trying GPU 0 as last resort...")
            gpu_0_info = EnhancedGPUForcer.check_gpu_memory_detailed(0)
            if gpu_0_info['free_gb'] >= min_free_gb / 2:  # Lower threshold for GPU 0
                return 0
        
        # Emergency fallback: try any GPU with cleanup
        logger.error("Emergency fallback: attempting cleanup on all GPUs...")
        for gpu_id in range(4):
            if EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id):
                final_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
                if final_info['free_gb'] >= 1.0:  # Very low threshold
                    logger.warning(f"Emergency: Using GPU {gpu_id} with {final_info['free_gb']:.1f}GB")
                    return gpu_id
        
        logger.error("No usable GPU found even after emergency cleanup")
        return None
    
    @staticmethod
    def force_gpu_environment_safe(gpu_id: int, cleanup_first: bool = True):
        """Safely force GPU environment with proper locking"""
        
        # Initialize locks if needed
        EnhancedGPUForcer.init_gpu_locks()
        
        # Acquire lock for this GPU
        with EnhancedGPUForcer._gpu_locks[gpu_id]:
            if cleanup_first:
                EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
            
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Conservative memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,"
                "max_split_size_mb:256,"
                "roundup_power2_divisions:16"
            )
            
            # Prevent memory fragmentation
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            
            # Set PyTorch device
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # 0 maps to our target GPU
                
            logger.info(f"✅ GPU environment forced to GPU {gpu_id}")
    
    @staticmethod
    def create_conservative_vllm_args(model_name: str, max_tokens: int = 4096):
        """Create very conservative VLLM engine args"""
        from vllm import AsyncEngineArgs
        
        return AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=min(max_tokens, 2048),  # Reduced max length
            gpu_memory_utilization=0.40,  # Very conservative
            device="cuda:0",
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs
            disable_log_stats=True,
            quantization=None,
            load_format="auto",
            dtype="float16",  # Use FP16 to save memory
            seed=42,
            swap_space=1,  # Minimal swap
            max_num_seqs=16,  # Small batch size
            max_num_batched_tokens=512,  # Very conservative
            max_seq_len_to_capture=512,  # Small sequences
            disable_custom_all_reduce=True,
            worker_use_ray=False,  # Disable Ray for stability
        )