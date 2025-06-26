# test_aggressive_gpu.py
"""
Test the aggressive GPU forcing mechanism
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gpu_force_fix import GPUForcer, test_gpu_forcing
from opulence_coordinator import initialize_dynamic_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_engine_creation():
    """Test creating an LLM engine with aggressive GPU forcing"""
    
    print("\n" + "="*60)
    print("TESTING AGGRESSIVE GPU ENGINE CREATION")
    print("="*60)
    
    # Step 1: Find best GPU
    best_gpu = GPUForcer.find_best_gpu_with_memory(2.0)  # Need at least 2GB
    
    if best_gpu is None:
        print("❌ No suitable GPU found")
        return False
    
    print(f"✅ Selected GPU {best_gpu}")
    
    # Step 2: Test coordinator initialization
    try:
        coordinator = await initialize_dynamic_system()
        print("✅ Coordinator initialized")
        
        # Step 3: Try to create an agent with the best GPU
        print(f"\n🔧 Creating agent on GPU {best_gpu}...")
        
        async with coordinator.get_agent_with_gpu("code_parser", preferred_gpu=best_gpu) as (agent, gpu_id):
            print(f"✅ Successfully created agent on GPU {gpu_id}")
            
            # Test that it's actually using the right GPU
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            print(f"📊 GPU {gpu_id} now has {memory_info['free_gb']:.1f}GB free")
            
            return True
            
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        
        # Show detailed error info
        import traceback
        traceback.print_exc()
        
        return False

async def test_file_processing():
    """Test actual file processing with forced GPU"""
    
    print("\n" + "="*60)
    print("TESTING FILE PROCESSING WITH FORCED GPU")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Find best GPU
        best_gpu = GPUForcer.find_best_gpu_with_memory(2.0)
        
        if best_gpu is None:
            print("❌ No suitable GPU found")
            return False
        
        print(f"🎯 Using GPU {best_gpu} for processing")
        
        # Test component analysis (this is what was failing)
        result = await coordinator.analyze_component(
            "TEST_COMPONENT", 
            "file", 
            preferred_gpu=best_gpu
        )
        
        if "error" not in result:
            print(f"✅ Successfully analyzed component!")
            print(f"   GPU used: {result.get('gpu_used', 'unknown')}")
            return True
        else:
            print(f"❌ Analysis failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ File processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_current_gpu_status():
    """Show current GPU status"""
    
    print("\n" + "="*60)
    print("CURRENT GPU STATUS")
    print("="*60)
    
    for gpu_id in range(4):
        memory_info = GPUForcer.check_gpu_memory(gpu_id)
        status = "🟢 Good" if memory_info['free_gb'] >= 8 else "🟡 Limited" if memory_info['free_gb'] >= 2 else "🔴 Poor"
        
        print(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free / {memory_info['total_gb']:.1f}GB total {status}")

async def main():
    """Main test function"""
    
    print("🚀 AGGRESSIVE GPU FORCING TEST")
    print("="*60)
    
    # Show current status
    show_current_gpu_status()
    
    # Test 1: Basic GPU forcing
    print(f"\n1️⃣ Testing basic GPU forcing...")
    if not test_gpu_forcing():
        print("❌ Basic GPU forcing failed")
        return
    
    # Test 2: Engine creation
    print(f"\n2️⃣ Testing engine creation...")
    if not await test_engine_creation():
        print("❌ Engine creation failed")
        return
    
    # Test 3: File processing
    print(f"\n3️⃣ Testing file processing...")
    if not await test_file_processing():
        print("❌ File processing failed")
        return
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print("The aggressive GPU forcing is working correctly!")

if __name__ == "__main__":
    asyncio.run(main())