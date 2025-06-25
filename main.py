# main.py
"""
Opulence Deep Research Mainframe Agent - Main Entry Point
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from opulence_coordinator import OpulenceCoordinator, OpulenceConfig
from utils.config_manager import config_manager
from utils.batch_processor import BatchProcessor

def setup_logging():
    """Setup logging configuration"""
    log_config = config_manager.get_section("logging")
    
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.get("file", "opulence.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def run_batch_processing(file_paths, coordinator):
    """Run batch processing mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing for {len(file_paths)} files")
    
    try:
        result = await coordinator.process_batch_files(file_paths)
        
        if result["status"] == "success":
            logger.info(f"Batch processing completed successfully")
            logger.info(f"Files processed: {result['files_processed']}")
            logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Print results summary
            print("\n" + "="*80)
            print("BATCH PROCESSING RESULTS")
            print("="*80)
            print(f"Status: {result['status']}")
            print(f"Files processed: {result['files_processed']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Print individual file results
            for i, file_result in enumerate(result.get("results", [])):
                if isinstance(file_result, dict):
                    print(f"\nFile {i+1}: {file_result.get('file_name', 'Unknown')}")
                    print(f"  Status: {file_result.get('status', 'Unknown')}")
                    if file_result.get('status') == 'success':
                        print(f"  Chunks created: {file_result.get('chunks_created', 0)}")
                        print(f"  File type: {file_result.get('file_type', 'Unknown')}")
                    elif 'error' in file_result:
                        print(f"  Error: {file_result['error']}")
            
            print("="*80)
            
        else:
            logger.error(f"Batch processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return 1
    
    return 0

async def run_analysis_mode(component_name, component_type, coordinator):
    """Run analysis mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis for {component_type or 'auto-detect'}: {component_name}")
    
    try:
        result = await coordinator.analyze_component(component_name, component_type)
        
        if "error" not in result:
            print("\n" + "="*80)
            print("COMPONENT ANALYSIS RESULTS")
            print("="*80)
            print(f"Component: {result.get('component_name', component_name)}")
            print(f"Type: {result.get('component_type', 'Unknown')}")
            
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
    
    print("Starting Opulence Web Interface...")
    print("Access the interface at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, cwd=project_root, env=env)
    except KeyboardInterrupt:
        print("\nShutting down web interface...")

def validate_files(file_paths):
    """Validate that files exist and are supported"""
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

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Opulence Deep Research Mainframe Agent")
    
    # Operation modes
    parser.add_argument("--mode", choices=["web", "batch", "analyze"], default="web",
                       help="Operation mode (default: web)")
    
    # Batch processing options
    parser.add_argument("--files", nargs="+", help="Files to process in batch mode")
    parser.add_argument("--folder", help="Folder containing files to process")
    
    # Analysis options
    parser.add_argument("--component", help="Component name for analysis")
    parser.add_argument("--type", choices=["field", "file", "table", "program", "jcl"],
                       help="Component type for analysis")
    
    # Configuration options
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--gpu-count", type=int, help="Number of GPUs to use")
    parser.add_argument("--max-time", type=int, help="Maximum processing time in seconds")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    if args.config:
        config_manager.config_file = Path(args.config)
        config_manager.load_config()
    
    # Override configuration with command line arguments
    if args.gpu_count:
        config_manager.set("system.gpu_count", args.gpu_count)
    
    if args.max_time:
        config_manager.set("system.max_processing_time", args.max_time)
    
    if args.log_level:
        config_manager.set("logging.level", args.log_level)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
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
        system_config = config_manager.get_section("system")
        config = OpulenceConfig(**system_config)
        coordinator = OpulenceCoordinator(config)
        await coordinator._init_agents()
        
        # Run batch processing
        return await run_batch_processing(valid_files, coordinator)
    
    elif args.mode == "analyze":
        if not args.component:
            logger.error("Analysis mode requires --component argument")
            return 1
        
        # Initialize coordinator
        system_config = config_manager.get_section("system")
        config = OpulenceConfig(**system_config)
        coordinator = OpulenceCoordinator(config)
        await coordinator._init_agents()
        
        # Run analysis
        return await run_analysis_mode(args.component, args.type, coordinator)
    
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













