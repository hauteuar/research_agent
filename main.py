# main.py
"""
Opulence Deep Research Mainframe Agent - Main Entry Point
Updated for Dynamic GPU Allocation System
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the new dynamic system
from opulence_coordinator import (
    DynamicOpulenceCoordinator, 
    OpulenceConfig,
    initialize_dynamic_system,
    get_dynamic_coordinator,
    update_system_config,
    get_current_config,
    optimize_gpu_assignments,
    backup_config
)
from utils.dynamic_config_manager import get_dynamic_config
from utils.batch_processor import BatchProcessor

def setup_logging():
    """Setup logging configuration using dynamic config manager"""
    config_manager = get_dynamic_config()
    log_config = config_manager.get_section("logging")
    
    # Create log handlers
    handlers = []
    
    # File handler
    log_file = log_config.get("file", "dynamic_opulence.log")
    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)
    
    # Console handler (unless quiet mode)
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized with dynamic configuration")

async def run_batch_processing(file_paths, coordinator):
    """Run batch processing mode with dynamic GPU allocation"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting dynamic batch processing for {len(file_paths)} files")
    
    # Show GPU status before processing
    health_status = coordinator.get_health_status()
    gpu_status = health_status.get('gpu_status', {})
    
    logger.info("GPU Status Before Processing:")
    for gpu_id, status in gpu_status.items():
        logger.info(f"  {gpu_id}: {status['utilization_percent']:.1f}% utilized, "
                   f"{status['memory_free_gb']:.1f}GB free")
    
    try:
        result = await coordinator.process_batch_files(file_paths)
        
        if result["status"] == "success":
            logger.info("Batch processing completed successfully")
            logger.info(f"Files processed: {result['files_processed']}")
            logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Print results summary
            print("\n" + "="*80)
            print("DYNAMIC BATCH PROCESSING RESULTS")
            print("="*80)
            print(f"Status: {result['status']}")
            print(f"Files processed: {result['files_processed']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"Dynamic GPU allocations: {coordinator.stats.get('dynamic_allocations', 0)}")
            print(f"Failed allocations: {coordinator.stats.get('failed_allocations', 0)}")
            
            # Print individual file results
            for i, file_result in enumerate(result.get("results", [])):
                if isinstance(file_result, dict):
                    print(f"\nFile {i+1}: {file_result.get('file_name', 'Unknown')}")
                    print(f"  Status: {file_result.get('status', 'Unknown')}")
                    print(f"  GPU Used: {file_result.get('gpu_used', 'Unknown')}")
                    
                    if file_result.get('status') == 'success':
                        print(f"  Chunks created: {file_result.get('chunks_created', 0)}")
                        print(f"  File type: {file_result.get('file_type', 'Unknown')}")
                    elif 'error' in file_result:
                        print(f"  Error: {file_result['error']}")
            
            # Show final GPU status
            final_health = coordinator.get_health_status()
            final_gpu_status = final_health.get('gpu_status', {})
            
            print("\nFinal GPU Status:")
            for gpu_id, status in final_gpu_status.items():
                print(f"  {gpu_id}: {status['utilization_percent']:.1f}% utilized, "
                     f"{status['active_workloads']} active workloads")
            
            print("="*80)
            
        else:
            logger.error(f"Batch processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return 1
    
    return 0

async def run_analysis_mode(component_name, component_type, coordinator, preferred_gpu=None):
    """Run analysis mode with dynamic GPU allocation"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting dynamic analysis for {component_type or 'auto-detect'}: {component_name}")
    
    if preferred_gpu is not None:
        logger.info(f"Preferred GPU: {preferred_gpu}")
    
    try:
        result = await coordinator.analyze_component(
            component_name, 
            component_type, 
            preferred_gpu=preferred_gpu
        )
        
        if "error" not in result:
            print("\n" + "="*80)
            print("DYNAMIC COMPONENT ANALYSIS RESULTS")
            print("="*80)
            print(f"Component: {result.get('component_name', component_name)}")
            print(f"Type: {result.get('component_type', 'Unknown')}")
            
            # Show GPU usage information
            gpu_used = result.get('gpu_used')
            gpus_used = result.get('gpus_used', [])
            
            if gpu_used is not None:
                print(f"GPU Used: {gpu_used}")
            elif gpus_used:
                print(f"GPUs Used: {gpus_used}")
            
            db2_gpu = result.get('db2_gpu_used')
            if db2_gpu is not None:
                print(f"DB2 Comparison GPU: {db2_gpu}")
            
            # Print summary based on component type
            if "lineage" in result:
                lineage = result["lineage"]
                usage_stats = lineage.get("usage_analysis", {}).get("statistics", {})
                print(f"Total references: {usage_stats.get('total_references', 0)}")
                print(f"Programs using: {len(usage_stats.get('programs_using', []))}")
                
                if "comprehensive_report" in lineage:
                    print("\nComprehensive Report:")
                    print("-" * 40)
                    print(lineage["comprehensive_report"])
            
            elif "logic_analysis" in result:
                logic = result["logic_analysis"]
                print(f"Total chunks: {logic.get('total_chunks', 0)}")
                print(f"Complexity score: {logic.get('complexity_score', 0):.2f}")
            
            # Show DB2 comparison if available
            if "db2_comparison" in result:
                db2_result = result["db2_comparison"]
                print(f"\nDB2 Comparison Status: {db2_result.get('status', 'Unknown')}")
            
            print("="*80)
            
        else:
            logger.error(f"Analysis failed: {result['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return 1
    
    return 0

async def run_web_interface():
    """Run Streamlit web interface"""
    import subprocess
    import os
    
    # Set environment variables for Streamlit
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Run Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
    
    print("Starting Opulence Dynamic GPU Web Interface...")
    print("Access the interface at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, cwd=project_root, env=env)
    except KeyboardInterrupt:
        print("\nShutting down web interface...")

def validate_files(file_paths):
    """Validate that files exist and are supported"""
    config_manager = get_dynamic_config()
    valid_files = []
    supported_extensions = config_manager.get("security.allowed_file_types", [])
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        if path.suffix.lower() not in supported_extensions:
            print(f"Warning: Unsupported file type: {file_path}")
            continue
        
        valid_files.append(path)
    
    return valid_files

async def run_gpu_status():
    """Show current GPU status"""
    coordinator = await initialize_dynamic_system()
    
    print("\n" + "="*80)
    print("GPU STATUS REPORT")
    print("="*80)
    
    health_status = coordinator.get_health_status()
    gpu_status = health_status.get('gpu_status', {})
    
    for gpu_id, status in gpu_status.items():
        print(f"\n{gpu_id.upper()}:")
        print(f"  Name: {status['name']}")
        print(f"  Status: {status['status']}")
        print(f"  Utilization: {status['utilization_percent']:.1f}%")
        print(f"  Memory Used: {status['memory_used_gb']:.1f}GB / {status['memory_total_gb']:.1f}GB")
        print(f"  Memory Free: {status['memory_free_gb']:.1f}GB")
        print(f"  Available: {'Yes' if status['is_available'] else 'No'}")
        print(f"  Active Workloads: {status['active_workloads']}")
    
    # Show agent preferences
    config_status = get_current_config()
    agent_mappings = config_status['agent_mappings']
    
    print(f"\nAGENT GPU PREFERENCES:")
    for agent, gpu in agent_mappings.items():
        status_text = f"GPU {gpu}" if gpu is not None else "Auto-select"
        print(f"  {agent}: {status_text}")
    
    print("="*80)

async def run_optimization():
    """Run GPU optimization"""
    coordinator = await initialize_dynamic_system()
    
    print("\n" + "="*80)
    print("GPU OPTIMIZATION")
    print("="*80)
    
    # Run optimization
    optimization_result = await coordinator.optimize_gpu_allocation()
    
    print(f"Optimization Status: {optimization_result['status']}")
    print(f"Suggestions Found: {optimization_result.get('total_suggestions', 0)}")
    
    for suggestion in optimization_result.get('optimization_suggestions', []):
        print(f"\n{suggestion['type'].upper()}:")
        print(f"  {suggestion['message']}")
        print(f"  Recommendation: {suggestion['recommendation']}")
    
    # Optimize agent assignments
    optimized_assignments = optimize_gpu_assignments()
    
    print(f"\nOptimized Agent Assignments:")
    for agent, gpu in optimized_assignments.items():
        print(f"  {agent}: GPU {gpu}")
    
    print("="*80)

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Opulence Deep Research Mainframe Agent (Dynamic GPU)")
    
    # Operation modes
    parser.add_argument("--mode", choices=["web", "batch", "analyze", "gpu-status", "optimize"], 
                       default="web", help="Operation mode (default: web)")
    
    # Batch processing options
    parser.add_argument("--files", nargs="+", help="Files to process in batch mode")
    parser.add_argument("--folder", help="Folder containing files to process")
    
    # Analysis options
    parser.add_argument("--component", help="Component name for analysis")
    parser.add_argument("--type", choices=["field", "file", "table", "program", "jcl"],
                       help="Component type for analysis")
    parser.add_argument("--preferred-gpu", type=int, choices=[0, 1, 2, 3],
                       help="Preferred GPU for analysis (0-3)")
    
    # Configuration options
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--gpu-count", type=int, help="Number of GPUs to use")
    parser.add_argument("--max-time", type=int, help="Maximum processing time in seconds")
    parser.add_argument("--memory-threshold", type=float, help="GPU memory threshold (0.1-1.0)")
    parser.add_argument("--utilization-threshold", type=float, help="GPU utilization threshold (10-100)")
    
    # Agent GPU preferences
    parser.add_argument("--set-agent-gpu", nargs=2, metavar=("AGENT", "GPU"),
                       help="Set GPU preference for agent (e.g., --set-agent-gpu code_parser 0)")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    # Configuration management
    parser.add_argument("--backup-config", action="store_true", help="Backup current configuration")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration")
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = get_dynamic_config()
    
    # Load custom configuration if provided
    if args.config:
        config_manager.config_file = Path(args.config)
        config_manager.load_config()
    
    # Override configuration with command line arguments
    config_updates = {}
    
    if args.gpu_count:
        config_updates["gpu.total_gpu_count"] = args.gpu_count
    
    if args.max_time:
        config_updates["system.max_processing_time"] = args.max_time
    
    if args.log_level:
        config_updates["logging.level"] = args.log_level
    
    if args.memory_threshold:
        config_updates["gpu.memory_threshold"] = args.memory_threshold
    
    if args.utilization_threshold:
        config_updates["gpu.utilization_threshold"] = args.utilization_threshold
    
    # Apply configuration updates
    if config_updates:
        for key, value in config_updates.items():
            config_manager.set(key, value)
    
    # Set agent GPU preference if specified
    if args.set_agent_gpu:
        agent_type, gpu_id = args.set_agent_gpu
        gpu_id = int(gpu_id) if gpu_id.isdigit() else None
        config_manager.set_agent_preferred_gpu(agent_type, gpu_id)
        print(f"Set {agent_type} preferred GPU to {gpu_id}")
    
    # Setup logging
    if not args.quiet:
        setup_logging()
    
    logger = logging.getLogger(__name__)
    
    # Handle configuration management commands
    if args.backup_config:
        backup_path = backup_config()
        print(f"Configuration backed up to: {backup_path}")
        return 0
    
    if args.validate_config:
        config_issues = config_manager.validate_config()
        if config_issues:
            print("Configuration issues found:")
            for issue in config_issues:
                print(f"  - {issue}")
            return 1
        else:
            print("Configuration validation passed")
            return 0
    
    # Validate configuration
    config_issues = config_manager.validate_config()
    if config_issues:
        logger.warning("Configuration issues found:")
        for issue in config_issues:
            logger.warning(f"  - {issue}")
    
    # Handle different modes
    if args.mode == "web":
        await run_web_interface()
        return 0
    
    elif args.mode == "gpu-status":
        await run_gpu_status()
        return 0
    
    elif args.mode == "optimize":
        await run_optimization()
        return 0
    
    elif args.mode == "batch":
        if not args.files and not args.folder:
            logger.error("Batch mode requires --files or --folder argument")
            return 1
        
        # Collect files to process
        file_paths = []
        
        if args.files:
            file_paths.extend(args.files)
        
        if args.folder:
            folder_path = Path(args.folder)
            if folder_path.exists() and folder_path.is_dir():
                supported_extensions = config_manager.get("security.allowed_file_types", [])
                for ext in supported_extensions:
                    file_paths.extend(folder_path.rglob(f"*{ext}"))
            else:
                logger.error(f"Folder not found: {args.folder}")
                return 1
        
        # Validate files
        valid_files = validate_files(file_paths)
        
        if not valid_files:
            logger.error("No valid files found to process")
            return 1
        
        # Initialize coordinator
        coordinator = await initialize_dynamic_system()
        
        # Run batch processing
        return await run_batch_processing(valid_files, coordinator)
    
    elif args.mode == "analyze":
        if not args.component:
            logger.error("Analysis mode requires --component argument")
            return 1
        
        # Initialize coordinator
        coordinator = await initialize_dynamic_system()
        
        # Run analysis
        return await run_analysis_mode(
            args.component, 
            args.type, 
            coordinator, 
            preferred_gpu=args.preferred_gpu
        )
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)