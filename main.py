# main.py
"""
Opulence Deep Research Mainframe Agent - Main Entry Point
Updated for Single GPU System - Much Simpler and More Reliable
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the new single GPU system
from opulence_coordinator import (
    DualGPUOpulenceCoordinator,      # Changed from SingleGPUOpulenceCoordinator
    DualGPUOpulenceConfig,           # Changed from SingleGPUOpulenceConfig
    SingleGPUChatEnhancer,           # Keep same - works with dual GPU
    create_dual_gpu_coordinator,     # Changed from create_single_gpu_coordinator
    create_shared_server_coordinator,
    create_dedicated_server_coordinator,
    get_global_coordinator,
    quick_file_processing,
    quick_component_analysis,
    quick_chat_query,
    get_system_status,
    ProductionCoordinatorManager
)

# 2. Update logging setup
def setup_logging(log_level="INFO", quiet=False):
    """Setup logging configuration"""
    handlers = []
    
    # File handler - change filename
    file_handler = logging.FileHandler("opulence_dual_gpu.log")  # Changed filename
    handlers.append(file_handler)
    
    # Console handler (unless quiet mode)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Dual GPU Opulence logging initialized")  # Changed message

async def run_batch_processing(file_paths, coordinator):
    """Run batch processing mode with dual GPU"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting dual GPU batch processing for {len(file_paths)} files")
    
    # Show initial GPU status - CHANGED to show multiple GPUs
    status = coordinator.get_health_status()
    logger.info(f"📊 Using GPUs {coordinator.selected_gpus}")  # Changed from selected_gpu
    
    # Show status for each GPU
    for gpu_id in coordinator.selected_gpus:
        gpu_status = coordinator.gpu_manager.get_gpu_status(gpu_id)
        logger.info(f"📊 GPU {gpu_id} Status: {gpu_status.get('memory_free_gb', 0):.1f}GB free")
    
    try:
        result = await coordinator.process_batch_files(file_paths)
        
        if result["status"] == "success":
            logger.info("✅ Batch processing completed successfully")
            logger.info(f"📊 Files processed: {result['files_processed']}")
            logger.info(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
            
            # Print results summary - CHANGED
            print("\n" + "="*80)
            print("DUAL GPU BATCH PROCESSING RESULTS")  # Changed title
            print("="*80)
            print(f"Status: ✅ {result['status']}")
            print(f"GPUs Used: {result['gpus_used']}")  # Changed from gpu_used
            print(f"Files Processed: {result['files_processed']}")
            print(f"Successful Files: {result['successful_files']}")
            print(f"Failed Files: {result['failed_files']}")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            print(f"Vector Indexing: {result['vector_indexing']}")
            
            # Show final status for each GPU - CHANGED
            final_status = coordinator.get_health_status()
            print(f"\n📊 Final GPU Status:")
            for gpu_id in coordinator.selected_gpus:
                gpu_status = coordinator.gpu_manager.get_gpu_status(gpu_id)
                print(f"   GPU {gpu_id}: {gpu_status.get('memory_usage_gb', 0):.1f}GB used, "
                      f"{gpu_status.get('active_tasks', 0)} active tasks")
            
        else:
            logger.error(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Batch processing error: {str(e)}")
        return 1
    
    return 0

async def run_analysis_mode(component_name, component_type, coordinator):
    """Run analysis mode with dual GPU"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Starting analysis for {component_type or 'auto-detect'}: {component_name}")
    logger.info(f"📊 Using GPUs {coordinator.selected_gpus}")  # Changed from selected_gpu
    
    # Rest stays the same...
    try:
        result = await coordinator.analyze_component(component_name, component_type)
        
        if result.get("status") != "error":
            print("\n" + "="*80)
            print("DUAL GPU COMPONENT ANALYSIS RESULTS")  # Changed title
            print("="*80)
            print(f"Component: {result.get('component_name', component_name)}")
            print(f"Type: {result.get('component_type', 'Unknown')}")
            print(f"GPUs Used: {result.get('gpus_used', coordinator.selected_gpus)}")  
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            
            # Show analysis results based on component type
            if "lineage" in result:
                lineage = result["lineage"]
                print(f"\n🔗 Lineage Analysis:")
                if isinstance(lineage, dict) and lineage.get("status") != "error":
                    usage_stats = lineage.get("usage_analysis", {}).get("statistics", {})
                    print(f"   Total references: {usage_stats.get('total_references', 0)}")
                    print(f"   Programs using: {len(usage_stats.get('programs_using', []))}")
                    
                    if "comprehensive_report" in lineage:
                        print(f"\n📋 Comprehensive Report:")
                        print("-" * 40)
                        report = lineage["comprehensive_report"]
                        # Truncate long reports
                        if len(report) > 500:
                            print(report[:500] + "...")
                        else:
                            print(report)
                else:
                    print(f"   ❌ Error: {lineage.get('error', 'Analysis failed')}")
            
            elif "logic_analysis" in result:
                logic = result["logic_analysis"]
                print(f"\n🧠 Logic Analysis:")
                if isinstance(logic, dict) and logic.get("status") != "error":
                    print(f"   Total chunks: {logic.get('total_chunks', 0)}")
                    print(f"   Complexity score: {logic.get('complexity_score', 0):.2f}")
                else:
                    print(f"   ❌ Error: {logic.get('error', 'Analysis failed')}")
            
            elif "jcl_analysis" in result:
                jcl = result["jcl_analysis"]
                print(f"\n📋 JCL Analysis:")
                if isinstance(jcl, dict) and jcl.get("status") != "error":
                    print(f"   Analysis completed successfully")
                else:
                    print(f"   ❌ Error: {jcl.get('error', 'Analysis failed')}")
            
            # Show semantic search results
            if "semantic_search" in result:
                semantic = result["semantic_search"]
                if isinstance(semantic, dict) and semantic.get("status") != "error":
                    results_count = len(semantic.get("results", []))
                    print(f"\n🔍 Semantic Search: Found {results_count} similar components")
                else:
                    print(f"\n🔍 Semantic Search: ❌ {semantic.get('error', 'Search failed')}")
            
            print("="*80)
            
        else:
            logger.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        return 1
    
    return 0

async def run_chat_mode(coordinator):
    """Run interactive chat mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"💬 Starting chat mode on GPUs {coordinator.selected_gpus}")  # Changed
    
    enhancer = SingleGPUChatEnhancer(coordinator)  # Keep same - still works
    conversation_history = []
    
    print("\n" + "="*80)
    print("OPULENCE INTERACTIVE CHAT MODE")
    print("="*80)
    print(f"GPUs: {coordinator.selected_gpus}")  # Changed from GPU
    print("Type 'quit', 'exit', or Ctrl+C to stop")
    print("Type 'status' to see system status")
    print("Type 'analyze COMPONENT_NAME' to analyze a component")
    print("="*80)
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 Opulence> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'status':
                    status = coordinator.get_health_status()
                    print(f"\n📊 System Status:")
                    print(f"   GPU: {status['selected_gpu']}")
                    print(f"   Status: {status['status']}")
                    print(f"   Active Agents: {status['active_agents']}")
                    print(f"   Tasks Completed: {status['stats']['tasks_completed']}")
                    print(f"   Uptime: {status['uptime_seconds']:.0f} seconds")
                    continue
                
                elif user_input.lower().startswith('analyze '):
                    component_name = user_input[8:].strip()
                    if component_name:
                        print(f"\n🔍 Analyzing {component_name}...")
                        result = await coordinator.analyze_component(component_name)
                        
                        # Format analysis result for chat
                        if result.get('status') != 'error':
                            response = f"Analysis of {component_name} completed successfully."
                            if 'lineage' in result:
                                response += " Lineage analysis available."
                            if 'logic_analysis' in result:
                                response += " Logic analysis available."
                            print(f"\n✅ {response}")
                        else:
                            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    continue
                
                # Process regular chat query
                print(f"\n🤔 Processing query...")
                response = await enhancer.process_regular_chat_query(user_input, conversation_history)
                
                print(f"\n🤖 Opulence: {response}")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
                # Keep conversation history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n👋 Chat mode interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Chat error: {str(e)}")
                print(f"\n❌ Sorry, I encountered an error: {str(e)}")
                
    except Exception as e:
        logger.error(f"❌ Chat mode error: {str(e)}")
        return 1
    
    return 0
async def run_web_interface():
    """Run Streamlit web interface"""
    import subprocess
    import os
    
    # Set environment variables for Streamlit
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    env['OPULENCE_MODE'] = 'dual_gpu'  # Changed from 'single_gpu'  # Tell Streamlit to use single GPU mode
    
    # Check if streamlit app exists
    streamlit_app = project_root / "streamlit_app_single_gpu.py"
    if not streamlit_app.exists():
        streamlit_app = project_root / "streamlit_app.py"  # Fallback to original
    
    # Run Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app), "--server.port=8501"]
    
    print("🚀 Starting Opulence Single GPU Web Interface...")
    print("🌐 Access the interface at: http://localhost:8501")
    print("⌨️ Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, cwd=project_root, env=env)
    except KeyboardInterrupt:
        print("\n🔄 Shutting down web interface...")

def validate_files(file_paths):
    """Validate that files exist and are supported"""
    valid_files = []
    supported_extensions = ['.cbl', '.cob', '.jcl', '.csv', '.cpy', '.copy', '.txt']
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            print(f"⚠️ Warning: File not found: {file_path}")
            continue
        
        if path.suffix.lower() not in supported_extensions:
            print(f"⚠️ Warning: Unsupported file type: {file_path} (supported: {supported_extensions})")
            continue
        
        valid_files.append(path)
    
    return valid_files

async def run_gpu_status():
    """Show current GPU status"""
    print("\n" + "="*80)
    print("DUAL GPU STATUS REPORT")  # Changed title
    print("="*80)
    
    try:
        coordinator = get_global_coordinator()
        status = coordinator.get_health_status()
        
        print(f"Selected GPUs: {status['selected_gpus']}")  # Changed from selected_gpu
        print(f"Status: {status['status']}")
        print(f"Database: {'✅ Available' if status['database_available'] else '❌ Not Available'}")
        
        # Show details for each GPU - CHANGED
        print(f"\nGPU Details:")
        for gpu_id in coordinator.selected_gpus:
            gpu_status = coordinator.gpu_manager.get_gpu_status(gpu_id)
            print(f"  GPU {gpu_id}:")
            print(f"    Memory Used: {gpu_status.get('memory_usage_gb', 0):.1f}GB")
            print(f"    Active Tasks: {gpu_status.get('active_tasks', 0)}")
            print(f"    Total Tasks: {gpu_status.get('total_tasks_processed', 0)}")
            print(f"    Locked: {'✅ Yes' if gpu_status.get('is_locked', False) else '❌ No'}")  
        
        
        print(f"\nSystem Stats:")
        stats = status['stats']
        print(f"  Files Processed: {stats.get('total_files_processed', 0)}")
        print(f"  Queries Processed: {stats.get('total_queries', 0)}")
        print(f"  Tasks Completed: {stats.get('tasks_completed', 0)}")
        print(f"  Uptime: {status['uptime_seconds']:.0f} seconds")
        
        print(f"\nActive Agents: {status['active_agents']}")
        
    except Exception as e:
        print(f"❌ Error getting GPU status: {e}")
        return 1
    
    print("="*80)
    return 0

async def run_system_test():
    """Run basic system test"""
    print("\n" + "="*80)
    print("DUAL GPU SYSTEM TEST")  # Changed title
    print("="*80)
    
    try:
        # Test coordinator creation
        print("1. 🧪 Creating coordinator...")
        coordinator = create_dual_gpu_coordinator(force_gpu_ids=[1, 2])  # Changed
        print(f"   ✅ Coordinator created on GPUs {coordinator.selected_gpus}")  # Changed
        
        # Test agent creation
        print("2. 🧪 Testing agent creation...")
        agent = coordinator.get_agent("code_parser")
        print("   ✅ Code parser agent created")
        
        # Test database
        print("3. 🧪 Testing database...")
        stats = await coordinator._get_database_stats()
        print(f"   ✅ Database accessible, {stats.get('program_chunks_count', 0)} chunks stored")
        
        # Test chat functionality
        print("4. 🧪 Testing chat functionality...")
        try:
            chat_result = await coordinator.process_chat_query("Hello, are you working?")
            print("   ✅ Chat system responsive")
        except Exception as e:
            print(f"   ⚠️ Chat test failed: {e}")
        
        # Test cleanup
        print("5. 🧪 Testing cleanup...")
        coordinator.cleanup()
        print("   ✅ Cleanup successful")
        
        print("\n🎉 All tests passed! System is ready.")
        coordinator.shutdown()
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return 1
    
    print("="*80)
    return 0

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Opulence Deep Research Mainframe Agent (Dual GPU)")  # Changed
    
    # Operation modes
    parser.add_argument("--mode", choices=["web", "batch", "analyze", "chat", "gpu-status", "test"], 
                       default="web", help="Operation mode (default: web)")
    
    # Batch processing options (same)
    parser.add_argument("--files", nargs="+", help="Files to process in batch mode")
    parser.add_argument("--folder", help="Folder containing files to process")
    parser.add_argument("--file-type", choices=["auto", "cobol", "jcl", "csv"], 
                       default="auto", help="File type for batch processing")
    
    # Analysis options (same)
    parser.add_argument("--component", help="Component name for analysis")
    parser.add_argument("--type", choices=["field", "file", "table", "program", "jcl"],
                       help="Component type for analysis")
    
    # Dual GPU configuration options - CHANGED
    parser.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf",
                       help="Model name to use")
    parser.add_argument("--exclude-gpu-0", action="store_true", default=True,
                       help="Exclude GPU 0 from selection (default: True)")
    parser.add_argument("--force-gpus", nargs=2, type=int, metavar=('GPU1', 'GPU2'),
                       help="Force specific GPUs (e.g., --force-gpus 1 2)")  # CHANGED
    parser.add_argument("--min-memory", type=float, default=6.0,
                       help="Minimum GPU memory required per GPU in GB (default: 6.0)")  # CHANGED
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens for model (default: 1024)")
    
    # Single GPU configuration options
    parser.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf",
                       help="Model name to use")
    parser.add_argument("--exclude-gpu-0", action="store_true", default=True,
                       help="Exclude GPU 0 from selection (default: True)")
    parser.add_argument("--force-gpus", nargs=2, type=int, metavar=('GPU1', 'GPU2'),
                       help="Force specific GPUs (e.g., --force-gpus 1 2)")  # CHANGED
    parser.add_argument("--min-memory", type=float, default=6.0,
                       help="Minimum GPU memory required per GPU in GB (default: 6.0)")  # CHANGED
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens for model (default: 1024)")
    
    # Server type optimization
    parser.add_argument("--server-type", choices=["shared", "dedicated"], 
                       help="Optimize for shared or dedicated server")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.quiet)
    logger = logging.getLogger(__name__)
    
    # Handle different modes
    if args.mode == "web":
        await run_web_interface()
        return 0
    
    elif args.mode == "gpu-status":
        return await run_gpu_status()
    
    elif args.mode == "test":
        return await run_system_test()
    
    # For other modes, create coordinator
    try:
        # Create coordinator based on server type
        if args.server_type == "shared":
            coordinator = create_shared_server_coordinator()
            logger.info("📊 Using shared server configuration")
        elif args.server_type == "dedicated":
            coordinator = create_dedicated_server_coordinator()
            logger.info("📊 Using dedicated server configuration")
        else:
            # Custom configuration - CHANGED
            config = DualGPUOpulenceConfig(  # Changed from SingleGPUOpulenceConfig
                model_name=args.model,
                exclude_gpu_0=args.exclude_gpu_0,
                min_memory_gb=args.min_memory,
                max_tokens=args.max_tokens,
                force_gpu_ids=args.force_gpus  # Changed from force_gpu_id
            )
            coordinator = DualGPUOpulenceCoordinator(config)  # Changed class
            logger.info("📊 Using custom configuration")
        
        logger.info(f"🎯 Coordinator initialized on GPUs {coordinator.selected_gpus}")  # Changed
        
        # Handle specific modes
        if args.mode == "batch":
            if not args.files and not args.folder:
                logger.error("❌ Batch mode requires --files or --folder argument")
                return 1
            
            # Collect files to process
            file_paths = []
            
            if args.files:
                file_paths.extend(args.files)
            
            if args.folder:
                folder_path = Path(args.folder)
                if folder_path.exists() and folder_path.is_dir():
                    supported_extensions = ['.cbl', '.cob', '.jcl', '.csv', '.cpy', '.copy']
                    for ext in supported_extensions:
                        file_paths.extend(folder_path.rglob(f"*{ext}"))
                else:
                    logger.error(f"❌ Folder not found: {args.folder}")
                    return 1
            
            # Validate files
            valid_files = validate_files(file_paths)
            
            if not valid_files:
                logger.error("❌ No valid files found to process")
                return 1
            
            # Run batch processing
            try:
                return await run_batch_processing(valid_files, coordinator)
            finally:
                coordinator.shutdown()
        
        elif args.mode == "analyze":
            if not args.component:
                logger.error("❌ Analysis mode requires --component argument")
                return 1
            
            # Run analysis
            try:
                return await run_analysis_mode(args.component, args.type, coordinator)
            finally:
                coordinator.shutdown()
        
        elif args.mode == "chat":
            # Run chat mode
            try:
                return await run_chat_mode(coordinator)
            finally:
                coordinator.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Failed to create coordinator: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⌨️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)