# test_updated_gpu_allocation.py
"""
Test the updated GPU allocation mechanism with new coordinator changes
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

async def test_gpu_availability_check():
    """Test the new GPU availability checking system"""
    
    print("\n" + "="*60)
    print("TESTING NEW GPU AVAILABILITY CHECKING")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Test the new availability status method
        gpu_status = coordinator.get_gpu_availability_status()
        
        print("📊 Detailed GPU Availability Status:")
        for gpu_key, status in gpu_status.items():
            gpu_id = gpu_key.split('_')[1]
            available = status.get('available', False)
            free_gb = status.get('free_memory_gb', 0)
            can_allocate = status.get('can_allocate', False)
            has_engine = status.get('has_llm_engine', False)
            
            status_icon = "✅" if available else "❌"
            alloc_icon = "🚀" if can_allocate else "🚫"
            engine_icon = "🧠" if has_engine else "💤"
            
            print(f"  GPU {gpu_id}: {status_icon} Available | {alloc_icon} Can Allocate | {engine_icon} Has Engine | {free_gb:.1f}GB free")
            
            if 'error' in status:
                print(f"    ⚠️  Error: {status['error']}")
        
        # Count available GPUs
        available_gpus = [k for k, v in gpu_status.items() if v.get('available', False)]
        print(f"\n📈 Summary: {len(available_gpus)} GPUs available for allocation")
        
        if len(available_gpus) >= 2:
            print("✅ GPU availability check passed - sufficient GPUs available")
            return True
        else:
            print("⚠️  Limited GPUs available - may impact performance")
            return len(available_gpus) > 0
            
    except Exception as e:
        print(f"❌ GPU availability check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_smart_gpu_allocation():
    """Test the new smart GPU allocation system"""
    
    print("\n" + "="*60)
    print("TESTING SMART GPU ALLOCATION")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Test 1: Get available GPU for agent (should use GPU manager)
        print("🔍 Testing get_available_gpu_for_agent...")
        
        best_gpu = await coordinator.get_available_gpu_for_agent("code_parser")
        if best_gpu is not None:
            print(f"✅ Smart allocation selected GPU {best_gpu} for code_parser")
        else:
            print("❌ No GPU available for allocation")
            return False
        
        # Test 2: Try preferred GPU allocation
        print(f"\n🎯 Testing preferred GPU allocation (requesting GPU {best_gpu})...")
        
        preferred_gpu = await coordinator.get_available_gpu_for_agent("data_loader", preferred_gpu=best_gpu)
        if preferred_gpu is not None:
            print(f"✅ Preferred allocation gave GPU {preferred_gpu}")
        else:
            print("❌ Preferred GPU allocation failed")
        
        # Test 3: Test fallback when preferred is unavailable
        print(f"\n🔄 Testing fallback allocation (requesting unavailable GPU 0)...")
        
        fallback_gpu = await coordinator.get_available_gpu_for_agent("lineage_analyzer", preferred_gpu=0)
        if fallback_gpu is not None:
            print(f"✅ Fallback allocation gave GPU {fallback_gpu}")
        else:
            print("❌ Fallback allocation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart GPU allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_creation_with_new_system():
    """Test agent creation with the new GPU allocation system"""
    
    print("\n" + "="*60)
    print("TESTING AGENT CREATION WITH NEW SYSTEM")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Test 1: Create agent with automatic GPU selection
        print("🤖 Creating code_parser agent with automatic GPU selection...")
        
        async with coordinator.get_agent_with_gpu("code_parser") as (agent, gpu_id):
            print(f"✅ Successfully created code_parser on GPU {gpu_id}")
            
            # Verify GPU usage
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            print(f"📊 GPU {gpu_id} memory after allocation: {memory_info['free_gb']:.1f}GB free")
            
            # Test 2: Create another agent (should use different GPU or share)
            print(f"\n🤖 Creating data_loader agent...")
            
            async with coordinator.get_agent_with_gpu("data_loader") as (agent2, gpu_id2):
                print(f"✅ Successfully created data_loader on GPU {gpu_id2}")
                
                if gpu_id != gpu_id2:
                    print(f"🎯 Good: Using different GPUs ({gpu_id} and {gpu_id2})")
                else:
                    print(f"🔄 Sharing GPU {gpu_id} between agents")
                
                # Test 3: Try with preferred GPU
                print(f"\n🎯 Creating logic_analyzer with preferred GPU...")
                
                async with coordinator.get_agent_with_gpu("logic_analyzer", preferred_gpu=gpu_id) as (agent3, gpu_id3):
                    print(f"✅ Successfully created logic_analyzer on GPU {gpu_id3}")
                    
                    if gpu_id3 == gpu_id:
                        print(f"🎯 Successfully used preferred GPU {gpu_id}")
                    else:
                        print(f"🔄 Fallback to GPU {gpu_id3} (preferred {gpu_id} not available)")
        
        print("✅ All agent creations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_cleanup_and_reload():
    """Test the new model cleanup and reload functionality"""
    
    print("\n" + "="*60)
    print("TESTING MODEL CLEANUP AND RELOAD")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Find a GPU to test with
        best_gpu = await coordinator.get_available_gpu_for_agent("test_agent")
        if best_gpu is None:
            print("❌ No GPU available for testing")
            return False
        
        print(f"🎯 Using GPU {best_gpu} for cleanup/reload test")
        
        # Step 1: Create an LLM engine
        print("🔧 Creating initial LLM engine...")
        engine1 = await coordinator.get_or_create_llm_engine(best_gpu)
        print(f"✅ Created engine: {type(engine1)}")
        
        # Check memory usage
        memory_before = GPUForcer.check_gpu_memory(best_gpu)
        print(f"📊 Memory before cleanup: {memory_before['free_gb']:.1f}GB free")
        
        # Step 2: Test cleanup
        print(f"\n🧹 Testing cleanup on GPU {best_gpu}...")
        await coordinator._cleanup_gpu_engine(best_gpu)
        
        # Check memory after cleanup
        memory_after = GPUForcer.check_gpu_memory(best_gpu)
        print(f"📊 Memory after cleanup: {memory_after['free_gb']:.1f}GB free")
        
        memory_freed = memory_after['free_gb'] - memory_before['free_gb']
        if memory_freed > 0:
            print(f"✅ Cleanup successful - freed {memory_freed:.1f}GB")
        else:
            print(f"⚠️  Cleanup may not have freed memory (diff: {memory_freed:.1f}GB)")
        
        # Step 3: Test reload
        print(f"\n🔄 Testing model reload on GPU {best_gpu}...")
        reload_success = await coordinator.reload_model_on_gpu(best_gpu)
        
        if reload_success:
            print("✅ Model reload successful")
            
            # Verify new engine works
            engine2 = await coordinator.get_or_create_llm_engine(best_gpu)
            print(f"✅ New engine created: {type(engine2)}")
            
            return True
        else:
            print("❌ Model reload failed")
            return False
            
    except Exception as e:
        print(f"❌ Cleanup/reload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_batch_file_processing():
    """Test batch file processing with the new GPU system"""
    
    print("\n" + "="*60)
    print("TESTING BATCH FILE PROCESSING WITH NEW GPU SYSTEM")
    print("="*60)
    
    try:
        coordinator = await initialize_dynamic_system()
        
        # Create some dummy file paths for testing
        test_files = [
            Path("test_file1.cbl"),
            Path("test_file2.csv"), 
            Path("test_file3.jcl"),
            Path("test_file4.cbl"),
            Path("test_file5.csv")
        ]
        
        print(f"🎯 Testing batch processing of {len(test_files)} files...")
        
        # Show GPU status before processing
        gpu_status_before = coordinator.get_gpu_availability_status()
        available_before = sum(1 for v in gpu_status_before.values() if v.get('available', False))
        print(f"📊 GPUs available before processing: {available_before}")
        
        # Note: This will fail because files don't exist, but we can test the GPU allocation logic
        result = await coordinator.process_batch_files(test_files)
        
        print(f"📋 Batch processing result: {result['status']}")
        print(f"📊 Dynamic allocations made: {coordinator.stats.get('dynamic_allocations', 0)}")
        print(f"📊 Failed allocations: {coordinator.stats.get('failed_allocations', 0)}")
        
        # Show GPU status after processing
        gpu_status_after = coordinator.get_gpu_availability_status()
        available_after = sum(1 for v in gpu_status_after.values() if v.get('available', False))
        print(f"📊 GPUs available after processing: {available_after}")
        
        if coordinator.stats.get('dynamic_allocations', 0) > 0:
            print("✅ GPU allocation system is working - allocations were made")
            return True
        else:
            print("⚠️  No GPU allocations made (may be due to file errors)")
            return False
            
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_enhanced_gpu_status():
    """Show enhanced GPU status with the new system"""
    
    print("\n" + "="*60)
    print("ENHANCED GPU STATUS")
    print("="*60)
    
    for gpu_id in range(4):
        try:
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            
            # Color coding based on available memory
            if memory_info['free_gb'] >= 8:
                status = "🟢 Excellent"
            elif memory_info['free_gb'] >= 4:
                status = "🟡 Good"
            elif memory_info['free_gb'] >= 2:
                status = "🟠 Limited"
            else:
                status = "🔴 Poor"
            
            print(f"GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free / {memory_info['total_gb']:.1f}GB total {status}")
            
        except Exception as e:
            print(f"GPU {gpu_id}: ❌ Error - {e}")

async def main():
    """Main test function for new GPU allocation system"""
    
    print("🚀 TESTING UPDATED GPU ALLOCATION SYSTEM")
    print("="*80)
    
    # Show current status
    show_enhanced_gpu_status()
    
    # Test 1: GPU availability checking
    print(f"\n1️⃣ Testing GPU availability checking...")
    test1_result = await test_gpu_availability_check()
    
    if not test1_result:
        print("❌ GPU availability check failed - stopping tests")
        return
    
    # Test 2: Smart GPU allocation
    print(f"\n2️⃣ Testing smart GPU allocation...")
    test2_result = await test_smart_gpu_allocation()
    
    # Test 3: Agent creation with new system
    print(f"\n3️⃣ Testing agent creation...")
    test3_result = await test_agent_creation_with_new_system()
    
    # Test 4: Model cleanup and reload
    print(f"\n4️⃣ Testing model cleanup and reload...")
    test4_result = await test_model_cleanup_and_reload()
    
    # Test 5: Batch processing
    print(f"\n5️⃣ Testing batch file processing...")
    test5_result = await test_batch_file_processing()
    
    # Summary
    print(f"\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    results = [
        ("GPU Availability Check", test1_result),
        ("Smart GPU Allocation", test2_result), 
        ("Agent Creation", test3_result),
        ("Model Cleanup/Reload", test4_result),
        ("Batch Processing", test5_result)
    ]
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The new GPU allocation system is working perfectly!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed! The system is mostly working correctly.")
    else:
        print("⚠️  Several tests failed. Check the logs above for issues.")

if __name__ == "__main__":
    asyncio.run(main())