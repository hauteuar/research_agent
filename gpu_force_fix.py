# gpu_force_fix_enhanced.py
"""
Enhanced GPU forcing mechanism with better memory detection, cleanup, and system resource monitoring
"""

import os
import torch
import logging
import subprocess
import time
import gc
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime as dt

logger = logging.getLogger(__name__)

class EnhancedGPUForcer:
    """Enhanced GPU forcing mechanism with advanced memory management and system monitoring"""
    
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Check overall system resource status"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Load average (Linux/Mac)
            try:
                load_avg = os.getloadavg()
                load_1min = load_avg[0]
            except (OSError, AttributeError):
                load_1min = 0.0
            
            # Check for critical conditions
            critical_load = (
                cpu_percent > 90 or 
                memory_percent > 95 or 
                load_1min > psutil.cpu_count() * 2
            )
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "load_1min": load_1min,
                "critical_load": critical_load,
                "timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "load_1min": 0,
                "critical_load": True,
                "error": str(e)
            }
    
    @staticmethod
    def check_gpu_memory_detailed(gpu_id: int) -> Dict[str, Any]:
        """Enhanced GPU memory checking with detailed process information"""
        try:
            # Primary memory query
            result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True, timeout=10)
            
            if not result.stdout.strip():
                raise RuntimeError(f"No output from nvidia-smi for GPU {gpu_id}")
            
            # Parse main metrics
            values = result.stdout.strip().split(',')
            if len(values) < 6:
                raise RuntimeError(f"Incomplete nvidia-smi output: {result.stdout}")
            
            total, used, free, util, temp, power = map(lambda x: float(x.strip()), values)
            
            # Get process information
            proc_result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-compute-apps=pid,used_memory,process_name',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            processes = []
            if proc_result.stdout.strip():
                for proc_line in proc_result.stdout.strip().split('\n'):
                    if proc_line.strip():
                        try:
                            parts = proc_line.split(',')
                            if len(parts) >= 2:
                                pid = int(parts[0].strip())
                                mem = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                                name = parts[2].strip() if len(parts) > 2 else "unknown"
                                processes.append({
                                    'pid': pid,
                                    'memory_mb': mem,
                                    'process_name': name
                                })
                        except (ValueError, IndexError):
                            continue
            
            # Calculate derived metrics
            free_gb = free / 1024
            total_gb = total / 1024
            used_gb = used / 1024
            memory_fragmentation = (used > 0 and free < 2048)  # Less than 2GB free but memory used
            
            # Determine availability
            is_available = (
                free_gb >= 1.5 and  # At least 1.5GB free
                util < 95 and       # Less than 95% utilization
                temp < 85 and       # Safe temperature
                len(processes) < 10  # Not too many processes
            )
            
            return {
                'total_mb': int(total),
                'used_mb': int(used),
                'free_mb': int(free),
                'free_gb': free_gb,
                'total_gb': total_gb,
                'used_gb': used_gb,
                'utilization_percent': util,
                'temperature': temp,
                'power_draw': power,
                'processes': processes,
                'process_count': len(processes),
                'is_fragmented': memory_fragmentation,
                'is_available': is_available,
                'gpu_id': gpu_id,
                'timestamp': dt.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"nvidia-smi timeout for GPU {gpu_id}")
            return EnhancedGPUForcer._get_error_gpu_status(gpu_id, "timeout")
        except subprocess.CalledProcessError as e:
            logger.error(f"nvidia-smi failed for GPU {gpu_id}: {e}")
            return EnhancedGPUForcer._get_error_gpu_status(gpu_id, f"nvidia-smi_error: {e}")
        except Exception as e:
            logger.error(f"GPU {gpu_id} memory check failed: {e}")
            return EnhancedGPUForcer._get_error_gpu_status(gpu_id, str(e))
    
    @staticmethod
    def _get_error_gpu_status(gpu_id: int, error: str) -> Dict[str, Any]:
        """Return error status for GPU"""
        return {
            'total_mb': 0, 'used_mb': 0, 'free_mb': 0,
            'free_gb': 0, 'total_gb': 0, 'used_gb': 0,
            'utilization_percent': 100, 'temperature': 0, 'power_draw': 0,
            'processes': [], 'process_count': 0,
            'is_fragmented': True, 'is_available': False,
            'gpu_id': gpu_id, 'error': error,
            'timestamp': dt.now().isoformat()
        }
    
    @staticmethod
    def find_optimal_gpu(min_free_gb: float = 2.0, exclude_gpu_0: bool = True, 
                        max_temp: float = 85.0) -> Optional[int]:
        """Find optimal GPU with comprehensive scoring"""
        best_gpu = None
        best_score = -1
        gpu_info = {}
        
        # Determine GPU range
        try:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            gpu_count = 0
        
        if gpu_count == 0:
            logger.error("No CUDA GPUs available")
            return None
        
        start_gpu = 1 if exclude_gpu_0 and gpu_count > 1 else 0
        
        for gpu_id in range(start_gpu, gpu_count):
            try:
                memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
                gpu_info[gpu_id] = memory_info
                
                free_gb = memory_info['free_gb']
                util = memory_info['utilization_percent']
                temp = memory_info['temperature']
                process_count = memory_info['process_count']
                
                logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free, {util:.1f}% util, "
                           f"{temp:.1f}°C, {process_count} processes")
                
                # Check basic requirements
                if (free_gb < min_free_gb or 
                    util > 95 or 
                    temp > max_temp or
                    not memory_info['is_available']):
                    continue
                
                # Calculate composite score
                score = EnhancedGPUForcer._calculate_gpu_score(memory_info, exclude_gpu_0)
                
                if score > best_score:
                    best_gpu = gpu_id
                    best_score = score
                    
            except Exception as e:
                logger.warning(f"Error evaluating GPU {gpu_id}: {e}")
                continue
        
        # Final check with fallback to GPU 0 if necessary
        if best_gpu is None and exclude_gpu_0 and gpu_count > 0:
            logger.warning("No suitable GPU found excluding GPU 0, checking GPU 0 as last resort...")
            try:
                memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(0)
                if memory_info['free_gb'] >= min_free_gb and memory_info['is_available']:
                    logger.warning(f"Using GPU 0 as last resort with {memory_info['free_gb']:.1f}GB free")
                    best_gpu = 0
            except Exception as e:
                logger.error(f"GPU 0 fallback check failed: {e}")
        
        if best_gpu is not None:
            logger.info(f"✅ Selected GPU {best_gpu} with score {best_score:.2f}")
        else:
            logger.error("❌ No suitable GPU found")
            logger.error("GPU Status Summary:")
            for gpu_id, info in gpu_info.items():
                logger.error(f"  GPU {gpu_id}: {info.get('free_gb', 0):.1f}GB free, "
                           f"available: {info.get('is_available', False)}")
        
        return best_gpu
    
    @staticmethod
    def _calculate_gpu_score(memory_info: Dict[str, Any], exclude_gpu_0: bool) -> float:
        """Calculate GPU suitability score"""
        score = 0.0
        
        # Free memory (most important factor)
        free_gb = memory_info['free_gb']
        score += free_gb * 30  # 30 points per GB
        
        # Lower utilization is better
        util = memory_info['utilization_percent']
        score += (100 - util) * 0.5
        
        # Lower temperature is better
        temp = memory_info['temperature']
        score += (100 - temp) * 0.3
        
        # Fewer processes is better
        proc_count = memory_info['process_count']
        score += max(0, (20 - proc_count) * 2)
        
        # Penalty for fragmentation
        if memory_info['is_fragmented']:
            score -= 10
        
        # Bonus for non-GPU-0 in shared systems
        if exclude_gpu_0 and memory_info['gpu_id'] > 0:
            score += 15
        
        # Power efficiency bonus (lower power draw)
        power = memory_info.get('power_draw', 0)
        if power > 0:
            score += max(0, (400 - power) * 0.1)  # Assume 400W max
        
        return max(score, 0.0)
    
    @staticmethod
    def aggressive_gpu_cleanup(gpu_id: int) -> bool:
        """Aggressive GPU cleanup with multiple strategies"""
        try:
            logger.info(f"🧹 Starting aggressive cleanup for GPU {gpu_id}")
            
            # Strategy 1: PyTorch cleanup
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                EnhancedGPUForcer.force_gpu_environment_safe(gpu_id, cleanup_first=False)
                
                if torch.cuda.is_available():
                    with torch.cuda.device(0):  # 0 maps to our target GPU
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Force garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Reset memory stats
                        try:
                            torch.cuda.reset_peak_memory_stats(0)
                            torch.cuda.reset_accumulated_memory_stats(0)
                        except Exception:
                            pass
                
            finally:
                # Restore environment
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Strategy 2: Wait for cleanup to take effect
            time.sleep(3)
            
            # Strategy 3: Verify cleanup worked
            memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
            cleanup_success = memory_info['free_gb'] > 1.0
            
            if cleanup_success:
                logger.info(f"✅ Cleanup successful for GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free")
            else:
                logger.warning(f"⚠️ Cleanup may have failed for GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free")
            
            return cleanup_success
            
        except Exception as e:
            logger.error(f"❌ Aggressive cleanup failed for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def cleanup_zombie_processes():
        """Check for zombie GPU processes (READ-ONLY for shared servers)"""
        try:
            logger.info("🔍 Checking GPU processes (read-only mode for shared server)")
            
            # Get all GPU processes
            result = subprocess.run([
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if not result.stdout.strip():
                logger.info("No GPU processes found")
                return
            
            active_processes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                try:
                    parts = line.split(',')
                    pid = int(parts[0].strip())
                    name = parts[2].strip() if len(parts) > 2 else "unknown"
                    memory = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                    
                    active_processes.append({
                        'pid': pid,
                        'name': name,
                        'memory_mb': memory
                    })
                        
                except (ValueError, IndexError):
                    continue
            
            if active_processes:
                logger.info(f"Found {len(active_processes)} active GPU processes:")
                for proc in active_processes[:5]:  # Show first 5
                    logger.info(f"  PID {proc['pid']}: {proc['name']} ({proc['memory_mb']}MB)")
                if len(active_processes) > 5:
                    logger.info(f"  ... and {len(active_processes) - 5} more")
            
        except Exception as e:
            logger.warning(f"GPU process check failed: {e}")
    
    @staticmethod
    def force_gpu_environment_safe(gpu_id: int, cleanup_first: bool = True):
        """Safely force GPU environment with optional cleanup"""
        try:
            if cleanup_first:
                # Basic cleanup first
                EnhancedGPUForcer.cleanup_zombie_processes()
                time.sleep(1)
            
            # Set CUDA environment
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Set memory management options
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            
            # Additional safety options
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Don't block CUDA calls
            
            # Set PyTorch device if available
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # 0 now maps to our target GPU
            
            logger.info(f"✅ Safely set GPU environment to GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to set GPU environment for GPU {gpu_id}: {e}")
            raise
    
    @staticmethod
    def create_conservative_vllm_args(model_name: str, max_tokens: int = 4096):
        """Create conservative VLLM engine args to avoid OOM"""
        try:
            from vllm import AsyncEngineArgs
            
            # Ensure max_num_batched_tokens is at least equal to max_model_len
            # Use the smaller of max_tokens or a conservative limit
            conservative_max_len = min(max_tokens, 2048)  # Cap at 2048 for memory efficiency
            max_batched_tokens = max(conservative_max_len, max_tokens)  # Must be >= max_model_len
            
            logger.info(f"vLLM config: max_model_len={conservative_max_len}, max_num_batched_tokens={max_batched_tokens}")
            
            return AsyncEngineArgs(
                model=model_name,
                tensor_parallel_size=1,
                max_model_len=conservative_max_len,  # Use conservative length
                gpu_memory_utilization=0.45,  # Very conservative
                device="cuda:0",
                trust_remote_code=True,
                enforce_eager=True,  # Disable CUDA graphs
                disable_log_stats=True,
                quantization=None,
                load_format="auto",
                dtype="auto",
                seed=42,
                swap_space=4,  # 4GB swap space
                max_num_seqs=16,  # Small batch size
                max_num_batched_tokens=max_batched_tokens,  # Must be >= max_model_len
                enable_prefix_caching=False,  # Disable to save memory
            )
            
        except ImportError:
            logger.error("vLLM not available, cannot create engine args")
            raise
        except Exception as e:
            logger.error(f"Failed to create vLLM args: {e}")
            raise
    
    @staticmethod
    def validate_gpu_environment(gpu_id: int) -> Dict[str, Any]:
        """Validate that GPU environment is correctly set up"""
        validation_result = {
            "gpu_id": gpu_id,
            "environment_vars": {},
            "cuda_available": False,
            "current_device": None,
            "memory_info": None,
            "validation_passed": False
        }
        
        try:
            # Check environment variables
            validation_result["environment_vars"] = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            }
            
            # Check CUDA availability
            validation_result["cuda_available"] = torch.cuda.is_available()
            
            if validation_result["cuda_available"]:
                validation_result["current_device"] = torch.cuda.current_device()
                
                # Get memory info from PyTorch perspective
                memory_info = torch.cuda.mem_get_info()
                validation_result["memory_info"] = {
                    "free_bytes": memory_info[0],
                    "total_bytes": memory_info[1],
                    "free_gb": memory_info[0] / (1024**3),
                    "total_gb": memory_info[1] / (1024**3)
                }
            
            # Validation checks
            checks_passed = 0
            total_checks = 4
            
            # Check 1: CUDA_VISIBLE_DEVICES is set correctly
            if os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_id):
                checks_passed += 1
            
            # Check 2: CUDA is available
            if validation_result["cuda_available"]:
                checks_passed += 1
            
            # Check 3: Current device is 0 (mapped to our GPU)
            if validation_result["current_device"] == 0:
                checks_passed += 1
            
            # Check 4: Has reasonable memory available
            if (validation_result["memory_info"] and 
                validation_result["memory_info"]["free_gb"] > 1.0):
                checks_passed += 1
            
            validation_result["validation_passed"] = (checks_passed >= 3)
            validation_result["checks_passed"] = f"{checks_passed}/{total_checks}"
            
            return validation_result
            
        except Exception as e:
            validation_result["error"] = str(e)
            return validation_result
    
    @staticmethod
    def emergency_gpu_reset(gpu_id: int) -> bool:
        """Emergency GPU reset - DISABLED for shared servers"""
        logger.warning(f"🚨 Emergency reset requested for GPU {gpu_id} - DISABLED on shared server")
        logger.info("In shared server environment, GPU reset could affect other users")
        logger.info("Instead, try: 1) Restart your process 2) Use different GPU 3) Contact admin")
        return False


def test_enhanced_gpu_forcer():
    """Test the enhanced GPU forcing mechanism"""
    print("Testing Enhanced GPU Forcing Mechanism")
    print("=" * 60)
    
    # System resource check
    print("1. Checking system resources...")
    system_status = EnhancedGPUForcer.check_system_resources()
    print(f"   CPU: {system_status['cpu_percent']:.1f}%")
    print(f"   Memory: {system_status['memory_percent']:.1f}%")
    print(f"   Load: {system_status['load_1min']:.2f}")
    print(f"   Critical: {system_status['critical_load']}")
    
    # Check all GPUs
    print("\n2. Checking all GPUs...")
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"   Detected {gpu_count} GPUs")
        
        for gpu_id in range(gpu_count):
            memory_info = EnhancedGPUForcer.check_gpu_memory_detailed(gpu_id)
            print(f"   GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free, "
                  f"{memory_info['utilization_percent']:.1f}% util, "
                  f"{memory_info['process_count']} processes, "
                  f"available: {memory_info['is_available']}")
    except Exception as e:
        print(f"   Error checking GPUs: {e}")
        return False
    
    # Find optimal GPU
    print("\n3. Finding optimal GPU...")
    optimal_gpu = EnhancedGPUForcer.find_optimal_gpu(min_free_gb=1.5, exclude_gpu_0=True)
    
    if optimal_gpu is None:
        print("   ❌ No suitable GPU found")
        return False
    
    print(f"   ✅ Found optimal GPU: {optimal_gpu}")
    
    # Test environment forcing
    print(f"\n4. Testing environment forcing for GPU {optimal_gpu}...")
    try:
        EnhancedGPUForcer.force_gpu_environment_safe(optimal_gpu, cleanup_first=True)
        print("   ✅ Environment forcing successful")
    except Exception as e:
        print(f"   ❌ Environment forcing failed: {e}")
        return False
    
    # Validate environment
    print(f"\n5. Validating GPU environment...")
    validation = EnhancedGPUForcer.validate_gpu_environment(optimal_gpu)
    print(f"   Validation passed: {validation['validation_passed']}")
    print(f"   Checks: {validation['checks_passed']}")
    print(f"   CUDA available: {validation['cuda_available']}")
    print(f"   Current device: {validation['current_device']}")
    
    if validation['memory_info']:
        print(f"   Memory available: {validation['memory_info']['free_gb']:.1f}GB")
    
    return validation['validation_passed']


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    success = test_enhanced_gpu_forcer()
    print(f"\nTest {'✅ PASSED' if success else '❌ FAILED'}")