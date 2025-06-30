#!/usr/bin/env python3
"""
Main Entry Point for Mainframe Deep Research Agent
Run this file to start the system with different options
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_streamlit_ui():
    """Launch the main Streamlit UI"""
    print("🚀 Starting Mainframe Analysis System UI...")
    print("🌐 Opening browser at http://localhost:8501")
    
    # Run streamlit with the main analysis interface
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "opulence_mainframe.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

def run_upload_ui():
    """Launch the upload-focused UI"""
    print("📤 Starting Data Upload System UI...")
    print("🌐 Opening browser at http://localhost:8502")
    
    # Run streamlit with the upload interface
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "opulence_data_upload.py",
        "--server.port=8502", 
        "--server.address=localhost"
    ])

def run_batch_analysis():
    """Run batch analysis from command line"""
    print("⚙️ Starting batch analysis...")
    
    # Import and run batch processing
    try:
        from opulence_mainframe import BatchProcessor, ReportGenerator
        
        # Get input directory
        input_dir = input("📁 Enter directory path with mainframe files: ").strip()
        
        if not os.path.exists(input_dir):
            print("❌ Directory not found!")
            return
        
        # Initialize processor
        processor = BatchProcessor()
        
        # Find mainframe files
        import glob
        patterns = ['*.cbl', '*.cob', '*.cobol', '*.jcl', '*.job', '*.cpy', '*.copy', '*.sql']
        all_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(input_dir, '**', pattern), recursive=True)
            all_files.extend(files)
        
        if not all_files:
            print("❌ No mainframe files found!")
            return
        
        print(f"📋 Found {len(all_files)} files")
        
        # Process files
        results = processor.batch_load_components(all_files)
        
        print(f"✅ Processed: {results['processed']}")
        print(f"❌ Errors: {results['errors']}")
        
        # Generate reports
        if results['processed'] > 0:
            report_gen = ReportGenerator()
            
            print("\n📊 Generating reports...")
            master_files = report_gen.generate_master_files_report()
            component_summary = report_gen.generate_component_summary_report()
            
            print(f"📈 Master files found: {len(master_files)}")
            print(f"📋 Components analyzed: {len(component_summary)}")
            print("\n💡 Run with --ui flag to see detailed analysis in browser")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

def run_llm_analysis():
    """Run LLM-powered intelligent analysis"""
    print("🧠 Starting LLM-powered analysis...")
    
    try:
        from opulence_mainframe import IntegratedMainframeAnalyzer, example_intelligent_analysis
        
        choice = input("Choose: (1) Example analysis (2) Custom analysis: ").strip()
        
        if choice == "1":
            example_intelligent_analysis()
        else:
            # Custom analysis
            input_dir = input("📁 Enter directory with mainframe files: ").strip()
            csv_files = input("📊 Enter CSV files directory (optional): ").strip()
            
            if not os.path.exists(input_dir):
                print("❌ Directory not found!")
                return
            
            # Initialize analyzer
            analyzer = IntegratedMainframeAnalyzer()
            
            # Process files (simplified version)
            import glob
            source_files = []
            for pattern in ['*.cbl', '*.cob', '*.jcl']:
                files = glob.glob(os.path.join(input_dir, pattern))
                source_files.extend(files)
            
            if source_files:
                components = []
                for file_path in source_files[:5]:  # Limit to 5 files for demo
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                    
                    components.append({
                        'name': os.path.basename(file_path),
                        'content': content,
                        'component_type': 'PROGRAM',
                        'analysis': {}
                    })
                
                csv_file_list = []
                if csv_files and os.path.exists(csv_files):
                    csv_file_list = glob.glob(os.path.join(csv_files, '*.csv'))
                
                # Run analysis
                results = analyzer.perform_comprehensive_analysis(components, csv_file_list)
                
                print(f"\n🎉 Analysis complete!")
                print(f"📋 Components analyzed: {len(results['component_analyses'])}")
                print(f"🎯 System recommendations: {len(results['system_recommendations'])}")
                print(f"⚠️ Overall risk level: {results['risk_assessment']['overall_risk_level']}")
                
                # Show sample insights
                for name, analysis in list(results['component_analyses'].items())[:2]:
                    print(f"\n📄 {name}:")
                    print(f"   Purpose: {analysis.business_purpose}")
                    print(f"   Risk: {analysis.change_risk_level}")
                    print(f"   Recommendations: {len(analysis.modernization_recommendations)}")
            
            else:
                print("❌ No source files found!")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 LLM analysis requires additional dependencies")

def show_menu():
    """Show interactive menu"""
    print("""
🖥️  Mainframe Deep Research Agent
=====================================

Choose how to run the system:

1. 🌐 Web Interface (Full System)     - Complete analysis with UI
2. 📤 Upload Interface                - Focus on data upload
3. ⚙️  Batch Analysis                 - Command-line processing  
4. 🧠 LLM Analysis                    - Intelligent analysis
5. 📊 Generate Reports                - Create analysis reports
6. ❓ Help & Documentation            - Show help information
7. 🚪 Exit

Enter your choice (1-7): """)

def generate_reports():
    """Generate reports from existing analysis"""
    print("📊 Generating reports from existing analysis...")
    
    try:
        from opulence_agent import ReportGenerator
        
        report_gen = ReportGenerator()
        
        print("\nAvailable reports:")
        print("1. Master Files Usage")
        print("2. Component Summary") 
        print("3. Field Usage Analysis")
        print("4. Unused Fields")
        print("5. Complexity Analysis")
        print("6. All Reports")
        
        choice = input("\nSelect report (1-6): ").strip()
        
        if choice == "1":
            df = report_gen.generate_master_files_report()
            print(f"\n📋 Master Files Report:")
            print(df.to_string(index=False))
        
        elif choice == "2":
            df = report_gen.generate_component_summary_report()
            print(f"\n📋 Component Summary:")
            print(df.to_string(index=False))
        
        elif choice == "3":
            df = report_gen.generate_field_usage_report()
            print(f"\n📋 Field Usage Report:")
            print(df.to_string(index=False))
        
        elif choice == "4":
            df = report_gen.generate_unused_fields_report()
            print(f"\n📋 Unused Fields Report:")
            if len(df) > 0:
                print(df.to_string(index=False))
            else:
                print("✅ No unused fields found!")
        
        elif choice == "5":
            df = report_gen.generate_complexity_report()
            print(f"\n📋 Complexity Analysis:")
            print(df.to_string(index=False))
        
        elif choice == "6":
            print("Generating all reports...")
            reports = {
                "Master Files": report_gen.generate_master_files_report(),
                "Component Summary": report_gen.generate_component_summary_report(),
                "Field Usage": report_gen.generate_field_usage_report(),
                "Unused Fields": report_gen.generate_unused_fields_report(),
                "Complexity": report_gen.generate_complexity_report()
            }
            
            for name, df in reports.items():
                print(f"\n📋 {name} Report:")
                if len(df) > 0:
                    print(df.head().to_string(index=False))
                    if len(df) > 5:
                        print(f"... and {len(df)-5} more rows")
                else:
                    print("No data found")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        print("💡 Make sure you have run analysis first")

def show_help():
    """Show help and documentation"""
    print("""
📚 Mainframe Deep Research Agent - Help
======================================

🎯 WHAT THIS SYSTEM DOES:
- Analyzes COBOL, JCL, DB2, CICS mainframe components
- Provides intelligent insights using LLM (CodeLlama)
- Compares file data with DB2 tables
- Generates migration roadmaps and recommendations
- Identifies master files, field usage, and system relationships

🚀 GETTING STARTED:

1. INSTALL DEPENDENCIES:
   pip install -r requirements.txt

2. PREPARE YOUR DATA:
   - COBOL programs (.cbl, .cob, .cobol)
   - JCL jobs (.jcl, .job)
   - Copybooks (.cpy, .copy)
   - SQL files (.sql, .ddl)
   - CSV data files (.csv)

3. RUN THE SYSTEM:
   python main.py                    # This interactive menu
   streamlit run mainframe_agent.py  # Direct UI launch

🌐 WEB INTERFACE FEATURES:
- Drag & drop file upload
- Batch processing of multiple files
- Interactive component analysis
- LLM-powered insights
- File vs DB2 comparison
- Migration roadmap generation
- Real-time chat with components

⚙️ COMMAND LINE FEATURES:
- Batch processing of entire directories
- Automated report generation
- Export results to CSV/JSON
- Integration with CI/CD pipelines

🧠 LLM INTELLIGENCE:
- Business purpose identification
- Complexity assessment
- Risk analysis with mitigation strategies
- Cross-component relationship analysis
- Intelligent modernization recommendations

📊 ANALYSIS OUTPUTS:
- Component summaries with metrics
- Master file usage reports
- Field lineage across programs
- Unused field identification
- Data quality assessments
- Migration priority rankings

🔗 INTEGRATION OPTIONS:
- REST API for external tools
- Database export/import
- FTP/SFTP file retrieval
- AWS S3 integration
- CI/CD pipeline integration

❓ NEED HELP?
- Check README.md for detailed documentation
- See example_usage.py for code examples
- Visit the web interface for interactive tutorials
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Mainframe Deep Research Agent')
    parser.add_argument('--ui', action='store_true', help='Launch web interface')
    parser.add_argument('--upload', action='store_true', help='Launch upload interface')
    parser.add_argument('--batch', action='store_true', help='Run batch analysis')
    parser.add_argument('--llm', action='store_true', help='Run LLM analysis')
    parser.add_argument('--reports', action='store_true', help='Generate reports')
    parser.add_argument('--help-docs', action='store_true', help='Show documentation')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.ui:
        run_streamlit_ui()
        return
    elif args.upload:
        run_upload_ui()
        return
    elif args.batch:
        run_batch_analysis()
        return
    elif args.llm:
        run_llm_analysis()
        return
    elif args.reports:
        generate_reports()
        return
    elif args.help_docs:
        show_help()
        return
    
    # Interactive menu if no arguments
    while True:
        show_menu()
        choice = input().strip()
        
        if choice == "1":
            run_streamlit_ui()
            break
        elif choice == "2":
            run_upload_ui()
            break
        elif choice == "3":
            run_batch_analysis()
        elif choice == "4":
            run_llm_analysis()
        elif choice == "5":
            generate_reports()
        elif choice == "6":
            show_help()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()

# Alternative simple launcher
def launch_ui():
    """Simple function to launch UI"""
    print("🚀 Launching Mainframe Analysis System...")
    run_streamlit_ui()

def launch_upload():
    """Simple function to launch upload interface"""
    print("📤 Launching Data Upload System...")
    run_upload_ui()

# Quick start functions
if __name__ == "__main__":
    # Check if running as module
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        launch_ui()
    elif len(sys.argv) > 1 and sys.argv[1] == "upload":
        launch_upload()
    else:
        main()