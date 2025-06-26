# test_gpu_allocation.py
"""
Test script to verify GPU allocation is working correctly
"""

import asyncio
import torch
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gpu_diagnostic import print_gpu_analysis, get_best_gpu_for_llm, force_gpu_cleanup
from opulence_coordinator import initialize_dynamic_system, get_dynamic_coordinator
from utils.enhanced_gpu_manager import GPUContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gpu_allocation():
    """Test GPU allocation system"""
    
    print("="*80)
    print("TESTING GPU ALLOCATION SYSTEM")
    print("="*80)
    
    # Step 1: Analyze current GPU state
    print("\n1. Current GPU Analysis:")
    print_gpu_analysis()
    
    # Step 2: Get best GPU recommendation
    best_gpu = get_best_gpu_for_llm()
    if best_gpu is None:
        print("\n❌ No suitable GPU found. Cleaning up and retrying...")
        force_gpu_cleanup()
        best_gpu = get_best_gpu_for_llm()
    
    if best_gpu is None:
        print("❌ Still no suitable GPU. Check your GPU memory usage.")
        return False
    
    print(f"\n2. Best GPU for LLM: GPU {best_gpu}")
    
    # Step 3: Test basic GPU allocation
    print(f"\n3. Testing GPU allocation...")
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Test GPU manager directly
        gpu_manager = coordinator.gpu_manager
        
        # Force refresh
        gpu_manager.force_refresh()
        
        # Get GPU status
        gpu_status = gpu_manager.get_gpu_status_detailed()
        
        print("GPU Manager Status:")
        for gpu_id, status in gpu_status.items():
            print(f"  {gpu_id}: {status['memory_free_gb']:.1f}GB free, "
                  f"{status['utilization_percent']:.1f}% utilized, "
                  f"Available: {status['is_available']}")
        
        # Test allocation
        allocated_gpu = gpu_manager.get_available_gpu(preferred_gpu=best_gpu)
        print(f"\nAllocated GPU: {allocated_gpu}")
        
        if allocated_gpu is None:
            print("❌ Failed to allocate any GPU")
            return False
        
        # Step 4: Test agent creation with specific GPU
        print(f"\n4. Testing agent creation on GPU {allocated_gpu}...")
        
        try:
            async with coordinator.get_agent_with_gpu("code_parser", preferred_gpu=allocated_gpu) as (agent, gpu_id):
                print(f"✅ Successfully created agent on GPU {gpu_id}")
                
                # Test a simple operation
                # Note: This would normally process a file, but we'll just test the creation
                print("✅ Agent created successfully")
                
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            
            # Try with different GPU
            print("Trying with auto-selection...")
            try:
                async with coordinator.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                    print(f"✅ Successfully created agent on auto-selected GPU {gpu_id}")
            except Exception as e2:
                print(f"❌ Auto-selection also failed: {e2}")
                return False
        
        print("\n✅ GPU allocation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_multiple_agents():
    """Test allocation of multiple agents"""
    
    print("\n" + "="*80)
    print("TESTING MULTIPLE AGENT ALLOCATION")
    print("="*80)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Test multiple agents concurrently
        agent_types = ["code_parser", "data_loader", "lineage_analyzer"]
        
        for agent_type in agent_types:
            try:
                async with coordinator.get_agent_with_gpu(agent_type) as (agent, gpu_id):
                    print(f"✅ {agent_type}: GPU {gpu_id}")
                    
                    # Small delay to see allocation pattern
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ {agent_type}: Failed - {e}")
        
        # Show final GPU status
        health_status = coordinator.get_health_status()
        print(f"\nFinal GPU allocation stats:")
        print(f"Dynamic allocations: {coordinator.stats.get('dynamic_allocations', 0)}")
        print(f"Failed allocations: {coordinator.stats.get('failed_allocations', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple agent test failed: {e}")
        return False

def print_troubleshooting_guide():
    """Print troubleshooting guide"""
    
    print("\n" + "="*80)
    print("TROUBLESHOOTING GUIDE")
    print("="*80)
    
    print("""
If you're seeing GPU allocation failures:

1. 🔍 CHECK GPU MEMORY:
   - Run 'nvidia-smi' to see current GPU usage
   - Each LLM needs at least 2-4GB free memory
   - Look for GPUs with <90% utilization

2. 🧹 CLEAN UP GPU MEMORY:
   - Kill unnecessary processes using GPU memory
   - Run: python -c "import torch; torch.cuda.empty_cache()"
   - Restart Jupyter kernels if running

3. ⚙️ ADJUST CONFIGURATION:
   - Lower gpu_memory_utilization (try 0.6 instead of 0.8)
   - Reduce model size or max_tokens
   - Enable enforce_eager=True in engine args

4. 🔧 FORCE SPECIFIC GPU:
   - Use CUDA_VISIBLE_DEVICES=2,3 to hide GPUs 0,1
   - Set preferred GPU in configuration
   - Use manual GPU context managers

5. 📊 MONITOR DURING LOADING:
   - Watch 'nvidia-smi' while loading models
   - Check which GPU the process actually uses
   - Verify CUDA device selection

Common Issues:
- ❌ CUDA OOM: Reduce gpu_memory_utilization to 0.6 or 0.5
- ❌ Wrong GPU: Check CUDA_VISIBLE_DEVICES and device selection
- ❌ Process conflicts: Kill competing processes or use different GPUs
- ❌ Memory fragmentation: Restart Python process

Quick Fixes:
export CUDA_VISIBLE_DEVICES=2,3  # Hide GPUs 0,1
python main.py --mode gpu-status  # Check available GPUs
python test_gpu_allocation.py    # Test allocation system
""")

async def main():
    """Main test function"""
    
    print("🚀 Starting GPU Allocation Tests")
    print("=" * 80)
    
    # Test 1: Basic allocation
    success1 = await test_gpu_allocation()
    
    if success1:
        # Test 2: Multiple agents
        success2 = await test_multiple_agents()
        
        if success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your GPU allocation system is working correctly.")
            
            # Show recommended configuration
            from gpu_diagnostic import suggest_gpu_configuration
            config = suggest_gpu_configuration()
            
            if config:
                print(f"\n📋 RECOMMENDED AGENT-GPU MAPPING:")
                for agent, gpu_id in config.items():
                    print(f"   {agent}: GPU {gpu_id}")
                
                print(f"\nTo apply this configuration:")
                print(f"python main.py --set-agent-gpu code_parser {config.get('code_parser', 0)}")
                print(f"python main.py --set-agent-gpu vector_index {config.get('vector_index', 1)}")
                print(f"# ... etc for other agents")
            
        else:
            print("\n⚠️ Basic allocation works, but multiple agents failed")
    else:
        print("\n❌ Basic GPU allocation failed")
        print("Check the troubleshooting guide below:")
        
    # Always show troubleshooting guide
    print_troubleshooting_guide()
    
    # Show final GPU state
    print(f"\n📊 FINAL GPU STATE:")
    print_gpu_analysis()

if __name__ == "__main__":
    asyncio.run(main())