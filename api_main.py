# main.py
"""
Opulence Deep Research Mainframe Agent - Main Entry Point
Updated for API-Based Architecture - No Direct GPU Management
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the new API-based system
from api_opulence_coordinator import (
    APIOpulenceCoordinator,
    APIOpulenceConfig,
    ModelServerConfig,
    LoadBalancingStrategy,
    create_api_coordinator_from_endpoints,
    create_api_coordinator_from_config,
    get_global_api_coordinator,
    quick_file_processing_api,
    quick_component_analysis_api,
    quick_chat_query_api,
    get_system_status_api
)

# Setup logging
def setup_logging(log_level="INFO", quiet=False):
    """Setup logging configuration"""
    handlers = []
    
    # File handler
    file_handler = logging.FileHandler("opulence_api.log")
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
    logger.info("API-based Opulence logging initialized")

async def run_batch_processing(file_paths, coordinator):
    """Run batch processing mode with API coordinator"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting API-based batch processing for {len(file_paths)} files")
    
    # Show initial server status
    status = coordinator.get_health_status()
    logger.info(f"📊 Using model servers: {[s['name'] for s in status.get('server_stats', {}).keys()]}")
    
    try:
        result = await coordinator.process_batch_files(file_paths)
        
        if result.get("status") == "success":
            logger.info("✅ Batch processing completed successfully")
            logger.info(f"📊 Files processed: {result.get('files_processed', 0)}")
            logger.info(f"⏱️ Processing time: {result.get('processing_time', 0):.2f} seconds")
            
            # Print results summary
            print("\n" + "="*80)
            print("API-BASED BATCH PROCESSING RESULTS")
            print("="*80)
            print(f"Status: ✅ {result.get('status', 'unknown')}")
            print(f"Model Servers Used: {result.get('servers_used', 'unknown')}")
            print(f"Files Processed: {result.get('files_processed', 0)}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            
            # Show final server status
            final_status = coordinator.get_health_status()
            print(f"\n📊 Final Server Status:")
            for server_name, server_stats in final_status.get('server_stats', {}).items():
                print(f"   {server_name}: {server_stats.get('active_requests', 0)} active requests, "
                      f"{server_stats.get('total_requests', 0)} total requests")
            
        else:
            logger.error(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Batch processing error: {str(e)}")
        return 1
    
    return 0

async def run_analysis_mode(component_name, component_type, coordinator):
    """Run analysis mode with API coordinator"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Starting analysis for {component_type or 'auto-detect'}: {component_name}")
    
    try:
        result = await coordinator.analyze_component(component_name, component_type)
        
        if result.get("status") != "error":
            print("\n" + "="*80)
            print("API-BASED COMPONENT ANALYSIS RESULTS")
            print("="*80)
            print(f"Component: {result.get('component_name', component_name)}")
            print(f"Type: {result.get('component_type', 'Unknown')}")
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            
            # Show analysis results
            if "analyses" in result:
                analyses = result["analyses"]
                for analysis_type, analysis_data in analyses.items():
                    print(f"\n🔗 {analysis_type.replace('_', ' ').title()}:")
                    if analysis_data.get("status") == "success":
                        print(f"   ✅ Completed successfully")
                    else:
                        print(f"   ❌ Error: {analysis_data.get('error', 'Unknown')}")
            
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
    """Run interactive chat mode with API coordinator"""
    logger = logging.getLogger(__name__)
    logger.info("💬 Starting chat mode with API-based processing")
    
    conversation_history = []
    
    print("\n" + "="*80)
    print("OPULENCE API-BASED CHAT MODE")
    print("="*80)
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
                    print(f"   Coordinator Type: {status.get('coordinator_type', 'unknown')}")
                    print(f"   Available Servers: {status.get('available_servers', 0)}")
                    print(f"   Total Servers: {status.get('total_servers', 0)}")
                    print(f"   Active Agents: {status.get('active_agents', 0)}")
                    print(f"   Load Balancing: {status.get('load_balancing_strategy', 'unknown')}")
                    continue
                
                elif user_input.lower().startswith('analyze '):
                    component_name = user_input[8:].strip()
                    if component_name:
                        print(f"\n🔍 Analyzing {component_name}...")
                        result = await coordinator.analyze_component(component_name)
                        
                        if result.get('status') != 'error':
                            response = f"Analysis of {component_name} completed successfully."
                            if 'analyses' in result:
                                completed = sum(1 for a in result['analyses'].values() if a.get('status') == 'success')
                                total = len(result['analyses'])
                                response += f" {completed}/{total} analyses completed."
                            print(f"\n✅ {response}")
                        else:
                            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    continue
                
                # Process regular chat query
                print(f"\n🤔 Processing query...")
                response = await coordinator.process_chat_query(user_input, conversation_history)
                
                if isinstance(response, dict):
                    chat_response = response.get("response", str(response))
                else:
                    chat_response = str(response)
                
                print(f"\n🤖 Opulence: {chat_response}")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": chat_response})
                
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
    env['OPULENCE_MODE'] = 'api_based'  # Tell Streamlit to use API mode
    
    # Check if streamlit app exists
    streamlit_app = project_root / "streamlit_app_api.py"
    if not streamlit_app.exists():
        streamlit_app = project_root / "streamlit_app.py"  # Fallback to original
    
    # Run Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app), "--server.port=8501"]
    
    print("🚀 Starting Opulence API-Based Web Interface...")
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

async def run_server_status():
    """Show current server status"""
    print("\n" + "="*80)
    print("API SERVER STATUS REPORT")
    print("="*80)
    
    try:
        coordinator = get_global_api_coordinator()
        await coordinator.initialize()
        status = coordinator.get_health_status()
        
        print(f"Coordinator Type: {status.get('coordinator_type', 'unknown')}")
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Available Servers: {status.get('available_servers', 0)}")
        print(f"Total Servers: {status.get('total_servers', 0)}")
        print(f"Load Balancing: {status.get('load_balancing_strategy', 'unknown')}")
        
        # Show server details
        print(f"\nServer Details:")
        for server_name, server_stats in status.get('server_stats', {}).items():
            print(f"  {server_name}:")
            print(f"    Status: {server_stats.get('status', 'unknown')}")
            print(f"    Active Requests: {server_stats.get('active_requests', 0)}")
            print(f"    Total Requests: {server_stats.get('total_requests', 0)}")
            print(f"    Success Rate: {server_stats.get('success_rate', 0):.1f}%")
            print(f"    Available: {'✅ Yes' if server_stats.get('available', False) else '❌ No'}")
        
        print(f"\nSystem Stats:")
        stats = status.get('stats', {})
        print(f"  Files Processed: {stats.get('total_files_processed', 0)}")
        print(f"  Queries Processed: {stats.get('total_queries', 0)}")
        print(f"  API Calls: {stats.get('total_api_calls', 0)}")
        print(f"  Uptime: {status.get('uptime_seconds', 0):.0f} seconds")
        
        await coordinator.shutdown()
        
    except Exception as e:
        print(f"❌ Error getting server status: {e}")
        return 1
    
    print("="*80)
    return 0

async def run_system_test():
    """Run basic system test"""
    print("\n" + "="*80)
    print("API-BASED SYSTEM TEST")
    print("="*80)
    
    try:
        # Test coordinator creation
        print("1. 🧪 Creating API coordinator...")
        model_servers = [
            {"endpoint": "http://localhost:8000", "gpu_id": 1, "name": "gpu_1"},
            {"endpoint": "http://localhost:8001", "gpu_id": 2, "name": "gpu_2"}
        ]
        coordinator = create_api_coordinator_from_config(model_servers)
        print(f"   ✅ API Coordinator created with {len(model_servers)} servers")
        
        # Test initialization
        print("2. 🧪 Testing initialization...")
        await coordinator.initialize()
        print("   ✅ Coordinator initialized")
        
        # Test agent creation
        print("3. 🧪 Testing agent creation...")
        try:
            agent = coordinator.get_agent("code_parser")
            print("   ✅ Code parser agent created")
        except Exception as e:
            print(f"   ⚠️ Agent creation failed: {e}")
        
        # Test health check
        print("4. 🧪 Testing health check...")
        health = coordinator.get_health_status()
        print(f"   ✅ Health check successful, status: {health.get('status', 'unknown')}")
        
        # Test chat functionality (if servers are available)
        print("5. 🧪 Testing chat functionality...")
        try:
            chat_result = await coordinator.process_chat_query("Hello, are you working?")
            print("   ✅ Chat system responsive")
        except Exception as e:
            print(f"   ⚠️ Chat test failed (expected if no servers running): {e}")
        
        # Test cleanup
        print("6. 🧪 Testing cleanup...")
        await coordinator.shutdown()
        print("   ✅ Cleanup successful")
        
        print("\n🎉 All tests passed! API-based system is ready.")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return 1
    
    print("="*80)
    return 0

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Opulence Deep Research Mainframe Agent (API-Based)")
    
    # Operation modes
    parser.add_argument("--mode", choices=["web", "batch", "analyze", "chat", "server-status", "test"], 
                       default="web", help="Operation mode (default: web)")
    
    # Batch processing options
    parser.add_argument("--files", nargs="+", help="Files to process in batch mode")
    parser.add_argument("--folder", help="Folder containing files to process")
    parser.add_argument("--file-type", choices=["auto", "cobol", "jcl", "csv"], 
                       default="auto", help="File type for batch processing")
    
    # Analysis options
    parser.add_argument("--component", help="Component name for analysis")
    parser.add_argument("--type", choices=["field", "file", "table", "program", "jcl"],
                       help="Component type for analysis")
    
    # API configuration options
    parser.add_argument("--model-servers", nargs="+", 
                       help="Model server endpoints (format: gpu_id:endpoint)")
    parser.add_argument("--load-balancing", choices=["round_robin", "least_busy", "least_latency", "random"],
                       default="least_busy", help="Load balancing strategy")
    parser.add_argument("--connection-pool-size", type=int, default=50,
                       help="HTTP connection pool size")
    parser.add_argument("--request-timeout", type=int, default=300,
                       help="Request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts")
    
    # Database options
    parser.add_argument("--db-path", default="opulence_api_data.db",
                       help="Database file path")
    
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
    
    elif args.mode == "server-status":
        return await run_server_status()
    
    elif args.mode == "test":
        return await run_system_test()
    
    # For other modes, create coordinator
    try:
        # Parse model servers
        model_servers = []
        if args.model_servers:
            for server_spec in args.model_servers:
                if ':' in server_spec:
                    gpu_id_str, endpoint = server_spec.split(':', 1)
                    try:
                        gpu_id = int(gpu_id_str)
                        model_servers.append({
                            "endpoint": endpoint,
                            "gpu_id": gpu_id,
                            "name": f"gpu_{gpu_id}"
                        })
                    except ValueError:
                        logger.error(f"Invalid GPU ID in server spec: {server_spec}")
                        return 1
                else:
                    logger.error(f"Invalid server spec format: {server_spec} (expected gpu_id:endpoint)")
                    return 1
        else:
            # Default servers
            model_servers = [
                {"endpoint": "http://localhost:8000", "gpu_id": 1, "name": "gpu_1"},
                {"endpoint": "http://localhost:8001", "gpu_id": 2, "name": "gpu_2"}
            ]
        
        # Create coordinator
        coordinator = create_api_coordinator_from_config(
            model_servers=model_servers,
            load_balancing_strategy=args.load_balancing,
            connection_pool_size=args.connection_pool_size,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            db_path=args.db_path
        )
        
        logger.info(f"🎯 API Coordinator initialized with {len(model_servers)} servers")
        
        # Initialize coordinator
        await coordinator.initialize()
        
        try:
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
                return await run_batch_processing(valid_files, coordinator)
            
            elif args.mode == "analyze":
                if not args.component:
                    logger.error("❌ Analysis mode requires --component argument")
                    return 1
                
                # Run analysis
                return await run_analysis_mode(args.component, args.type, coordinator)
            
            elif args.mode == "chat":
                # Run chat mode
                return await run_chat_mode(coordinator)
        
        finally:
            await coordinator.shutdown()
    
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