# gpu_diagnostic.py
"""
GPU Diagnostic and Selection Tool for Opulence
"""

import torch
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)

def get_nvidia_smi_info() -> Dict[str, any]:
    """Get detailed GPU information from nvidia-smi"""
    try:
        # Get GPU info in JSON format
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total_mb': int(parts[2]),
                        'memory_used_mb': int(parts[3]),
                        'memory_free_mb': int(parts[4]),
                        'utilization_percent': float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                        'temperature_c': float(parts[6]) if parts[6] != '[Not Supported]' else 0.0,
                        'power_draw_w': float(parts[7]) if parts[7] != '[Not Supported]' else 0.0
                    })
        
        # Get process information
        proc_result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        processes = []
        for line in proc_result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    processes.append({
                        'gpu_uuid': parts[0],
                        'pid': parts[1],
                        'process_name': parts[2],
                        'used_memory_mb': int(parts[3])
                    })
        
        return {'gpus': gpus, 'processes': processes}
        
    except Exception as e:
        logger.error(f"Failed to get nvidia-smi info: {e}")
        return {'gpus': [], 'processes': []}

def analyze_gpu_suitability() -> List[Dict[str, any]]:
    """Analyze GPU suitability for LLM workloads"""
    info = get_nvidia_smi_info()
    gpus = info['gpus']
    
    recommendations = []
    
    for gpu in gpus:
        gpu_id = gpu['index']
        free_gb = gpu['memory_free_mb'] / 1024
        total_gb = gpu['memory_total_mb'] / 1024
        used_gb = gpu['memory_used_mb'] / 1024
        utilization = gpu['utilization_percent']
        
        # Determine suitability
        suitability = "unsuitable"
        reason = ""
        priority = 0
        
        if free_gb >= 8.0:  # Excellent for LLM
            suitability = "excellent"
            reason = f"Abundant memory: {free_gb:.1f}GB free"
            priority = 100 - utilization
        elif free_gb >= 4.0:  # Good for LLM
            suitability = "good"
            reason = f"Good memory: {free_gb:.1f}GB free"
            priority = 80 - utilization
        elif free_gb >= 2.0:  # Marginal for LLM
            suitability = "marginal"
            reason = f"Limited memory: {free_gb:.1f}GB free"
            priority = 60 - utilization
        elif free_gb >= 1.0:  # Poor for LLM
            suitability = "poor"
            reason = f"Very limited memory: {free_gb:.1f}GB free"
            priority = 40 - utilization
        else:  # Unsuitable
            suitability = "unsuitable"
            reason = f"Insufficient memory: {free_gb:.1f}GB free (need at least 1GB)"
            priority = 0
        
        # Adjust priority based on utilization
        if utilization > 90:
            suitability = "busy"
            reason += f", high utilization: {utilization:.1f}%"
            priority = max(0, priority - 50)
        elif utilization > 50:
            reason += f", moderate utilization: {utilization:.1f}%"
            priority = max(0, priority - 20)
        
        recommendations.append({
            'gpu_id': gpu_id,
            'name': gpu['name'],
            'suitability': suitability,
            'reason': reason,
            'priority': priority,
            'memory_free_gb': free_gb,
            'memory_total_gb': total_gb,
            'memory_used_gb': used_gb,
            'utilization_percent': utilization,
            'temperature_c': gpu['temperature_c'],
            'power_draw_w': gpu['power_draw_w']
        })
    
    # Sort by priority (highest first)
    recommendations.sort(key=lambda x: x['priority'], reverse=True)
    
    return recommendations

def get_best_gpu_for_llm() -> Optional[int]:
    """Get the best GPU ID for LLM workload"""
    recommendations = analyze_gpu_suitability()
    
    for gpu in recommendations:
        if gpu['suitability'] in ['excellent', 'good', 'marginal'] and gpu['memory_free_gb'] >= 2.0:
            logger.info(f"Selected GPU {gpu['gpu_id']} for LLM: {gpu['reason']}")
            return gpu['gpu_id']
    
    logger.warning("No suitable GPU found for LLM workload")
    return None

def print_gpu_analysis():
    """Print detailed GPU analysis"""
    print("\n" + "="*80)
    print("GPU DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    recommendations = analyze_gpu_suitability()
    
    if not recommendations:
        print("No GPUs found or nvidia-smi not available")
        return
    
    print(f"{'GPU':<3} {'Name':<25} {'Memory (Used/Total)':<20} {'Free':<8} {'Util':<6} {'Suitability':<12} {'Reason'}")
    print("-" * 100)
    
    for gpu in recommendations:
        memory_str = f"{gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB"
        print(f"{gpu['gpu_id']:<3} {gpu['name'][:24]:<25} {memory_str:<20} "
              f"{gpu['memory_free_gb']:.1f}GB{'':<2} {gpu['utilization_percent']:.1f}%{'':<2} "
              f"{gpu['suitability']:<12} {gpu['reason']}")
    
    print("-" * 100)
    
    # Recommendations
    best_gpus = [gpu for gpu in recommendations if gpu['suitability'] in ['excellent', 'good']]
    marginal_gpus = [gpu for gpu in recommendations if gpu['suitability'] == 'marginal']
    
    if best_gpus:
        print(f"\n✅ RECOMMENDED GPUs for LLM: {', '.join([str(gpu['gpu_id']) for gpu in best_gpus])}")
        print(f"   Best choice: GPU {best_gpus[0]['gpu_id']} ({best_gpus[0]['reason']})")
    elif marginal_gpus:
        print(f"\n⚠️  MARGINAL GPUs (may work): {', '.join([str(gpu['gpu_id']) for gpu in marginal_gpus])}")
        print(f"   Try: GPU {marginal_gpus[0]['gpu_id']} ({marginal_gpus[0]['reason']})")
    else:
        print("\n❌ NO SUITABLE GPUs found for LLM workload")
        print("   All GPUs are either too busy or have insufficient memory")
    
    print("="*80)

def force_gpu_cleanup():
    """Force cleanup of GPU memory"""
    if torch.cuda.is_available():
        print("Forcing GPU memory cleanup...")
        
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"  GPU {i}: Memory cleared")
            except Exception as e:
                print(f"  GPU {i}: Cleanup failed - {e}")
        
        print("GPU memory cleanup completed")
    else:
        print("CUDA not available")

def suggest_gpu_configuration() -> Dict[str, int]:
    """Suggest optimal GPU configuration for agents"""
    recommendations = analyze_gpu_suitability()
    
    # Filter suitable GPUs
    suitable_gpus = [
        gpu for gpu in recommendations 
        if gpu['suitability'] in ['excellent', 'good', 'marginal'] and gpu['memory_free_gb'] >= 2.0
    ]
    
    if not suitable_gpus:
        return {}
    
    # Agent memory requirements (rough estimates)
    agents = [
        ('code_parser', 'medium'),      # 3-4GB
        ('vector_index', 'high'),       # 4-6GB  
        ('lineage_analyzer', 'medium'), # 3-4GB
        ('logic_analyzer', 'medium'),   # 3-4GB
        ('data_loader', 'low'),         # 2-3GB
        ('documentation', 'low'),       # 2-3GB
        ('db2_comparator', 'medium')    # 3-4GB
    ]
    
    # Sort agents by memory requirements
    high_mem_agents = [agent for agent, req in agents if req == 'high']
    med_mem_agents = [agent for agent, req in agents if req == 'medium']
    low_mem_agents = [agent for agent, req in agents if req == 'low']
    
    assignments = {}
    gpu_index = 0
    
    # Assign high memory agents to best GPUs
    for agent in high_mem_agents:
        if gpu_index < len(suitable_gpus):
            assignments[agent] = suitable_gpus[gpu_index]['gpu_id']
            gpu_index += 1
    
    # Assign medium memory agents
    for agent in med_mem_agents:
        if gpu_index < len(suitable_gpus):
            assignments[agent] = suitable_gpus[gpu_index]['gpu_id']
            gpu_index += 1
        else:
            # Reuse GPUs if we've run out
            assignments[agent] = suitable_gpus[gpu_index % len(suitable_gpus)]['gpu_id']
            gpu_index += 1
    
    # Assign low memory agents
    for agent in low_mem_agents:
        assignments[agent] = suitable_gpus[gpu_index % len(suitable_gpus)]['gpu_id']
        gpu_index += 1
    
    return assignments

if __name__ == "__main__":
    print_gpu_analysis()
    
    print("\nSuggested Agent GPU Configuration:")
    config = suggest_gpu_configuration()
    for agent, gpu_id in config.items():
        print(f"  {agent}: GPU {gpu_id}")
    
    best_gpu = get_best_gpu_for_llm()
    if best_gpu is not None:
        print(f"\nBest GPU for immediate LLM use: GPU {best_gpu}")
    
    print(f"\nTo use the recommended GPU in your code:")
    print(f"coordinator = await initialize_dynamic_system()")
    print(f"# Then manually set preferred GPU or let system auto-select")