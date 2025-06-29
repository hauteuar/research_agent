import subprocess
import psutil
import time
import os
import signal

def emergency_cleanup():
    print("🚨 Starting emergency cleanup...")
    
    # 1. Kill zombie Python processes
    for proc in psutil.process_iter(['pid', 'name', 'status', 'cmdline']):
        try:
            if (proc.info['status'] == psutil.STATUS_ZOMBIE and 
                'python' in proc.info['name'].lower()):
                print(f"Killing zombie process {proc.info['pid']}")
                proc.kill()
        except:
            continue
    
    # 2. Kill hanging CUDA/vLLM processes
    subprocess.run(['pkill', '-f', 'vllm'], capture_output=True)
    subprocess.run(['pkill', '-f', 'cuda'], capture_output=True)
    
    # 3. Reset all GPUs
    for gpu_id in range(4):
        try:
            subprocess.run(['nvidia-smi', '--gpu-reset', f'-i', str(gpu_id)], 
                         capture_output=True, timeout=10)
            print(f"Reset GPU {gpu_id}")
        except:
            continue
    
    # 4. Clear system memory
    subprocess.run(['sync'], capture_output=True)
    with open('/proc/sys/vm/drop_caches', 'w') as f:
        f.write('3')
    
    print("✅ Emergency cleanup completed")

if __name__ == "__main__":
    emergency_cleanup()