# gpu_force_fix.py
"""
Aggressive GPU forcing mechanism for VLLM/LLM engines
"""

import os
import torch
import logging
from typing import Optional
import subprocess

logger = logging.getLogger(__name__)

class GPUForcer:
    """Force GPU selection for LLM engines"""
    
    @staticmethod
    def force_gpu_environment(gpu_id: int):
        """Aggressively force GPU environment"""
        
        # Method 1: Set CUDA_VISIBLE_DEVICES to only show target GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # Method 2: Set PyTorch CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Now 0 maps to our target GPU
            logger.info(f"Set torch.cuda.set_device(0) -> actual GPU {gpu_id}")
        
        # Method 3: Set additional CUDA environment variables
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Method 4: Force clean up any existing CUDA context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def check_gpu_memory(gpu_id: int) -> dict:
        """Check actual GPU memory using nvidia-ml-py or nvidia-smi"""
        try:
            # Use nvidia-smi to get accurate memory info
            result = subprocess.run([
                'nvidia-smi', 
                f'--id={gpu_id}',
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            line = result.stdout.strip()
            if line:
                total, used, free = map(int, line.split(','))
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'free_gb': free / 1024,
                    'total_gb': total / 1024
                }
        except Exception as e:
            logger.error(f"Failed to get GPU {gpu_id} memory: {e}")
        
        return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'free_gb': 0, 'total_gb': 0}
    
    @staticmethod
    def find_best_gpu_with_memory(min_free_gb: float = 8.0) -> Optional[int]:
        """Find GPU with sufficient free memory"""
        best_gpu = None
        best_free_memory = 0
        
        for gpu_id in range(4):  # Check GPUs 0-3
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            free_gb = memory_info['free_gb']
            
            logger.info(f"GPU {gpu_id}: {free_gb:.1f}GB free")
            
            if free_gb >= min_free_gb and free_gb > best_free_memory:
                best_gpu = gpu_id
                best_free_memory = free_gb
        
        if best_gpu is not None:
            logger.info(f"Selected GPU {best_gpu} with {best_free_memory:.1f}GB free")
        else:
            logger.warning(f"No GPU found with at least {min_free_gb}GB free")
            
            # Try with lower requirement
            for gpu_id in range(4):
                memory_info = GPUForcer.check_gpu_memory(gpu_id)
                free_gb = memory_info['free_gb']
                
                if free_gb >= 2.0 and free_gb > best_free_memory:  # Lower requirement
                    best_gpu = gpu_id
                    best_free_memory = free_gb
            
            if best_gpu is not None:
                logger.info(f"Selected GPU {best_gpu} with {best_free_memory:.1f}GB free (lower threshold)")
        
        return best_gpu
    
    @staticmethod
    def create_vllm_engine_args(model_name: str, max_tokens: int = 4096):
        """Create VLLM engine args with aggressive GPU forcing"""
        from vllm import AsyncEngineArgs
        
        # The GPU is now mapped to device 0 due to CUDA_VISIBLE_DEVICES
        return AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_tokens,
            gpu_memory_utilization=0.65,  # Conservative memory usage
            device="cuda:0",  # This is now our target GPU
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs
            disable_log_stats=True,
            quantization=None,
            load_format="auto",
            dtype="auto",
            seed=42
        )

def test_gpu_forcing():
    """Test the GPU forcing mechanism"""
    print("Testing GPU Forcing Mechanism")
    print("=" * 50)
    
    # Find best GPU
    best_gpu = GPUForcer.find_best_gpu_with_memory(8.0)
    
    if best_gpu is None:
        print("❌ No suitable GPU found")
        return False
    
    print(f"✅ Found suitable GPU: {best_gpu}")
    
    # Force GPU environment
    print(f"🔧 Forcing GPU environment to use GPU {best_gpu}")
    GPUForcer.force_gpu_environment(best_gpu)
    
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

if __name__ == "__main__":
    test_gpu_forcing()