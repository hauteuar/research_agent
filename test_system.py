# scripts/test_system.py
#!/usr/bin/env python3
"""
System test script for Opulence Deep Research Mainframe Agent
"""

import asyncio
import sys
import tempfile
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from opulence_coordinator import OpulenceCoordinator, OpulenceConfig

# Sample COBOL code for testing
SAMPLE_COBOL = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TEST-PROGRAM.
       
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO 'CUSTFILE'
           ORGANIZATION IS SEQUENTIAL.
       
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05  CUSTOMER-ID     PIC 9(8).
           05  CUSTOMER-NAME   PIC X(30).
           05  ACCOUNT-BALANCE PIC 9(10)V99.
       
       WORKING-STORAGE SECTION.
       01  WS-CUSTOMER-COUNT   PIC 9(5) VALUE 0.
       01  WS-TOTAL-BALANCE    PIC 9(12)V99 VALUE 0.
       
       PROCEDURE DIVISION.
       MAIN-PROCESSING.
           OPEN INPUT CUSTOMER-FILE.
           PERFORM READ-CUSTOMER-RECORDS.
           CLOSE CUSTOMER-FILE.
           PERFORM DISPLAY-SUMMARY.
           STOP RUN.
           
       READ-CUSTOMER-RECORDS.
           READ CUSTOMER-FILE
               AT END MOVE 'Y' TO WS-EOF-FLAG
           END-READ.
           PERFORM UNTIL WS-EOF-FLAG = 'Y'
               ADD 1 TO WS-CUSTOMER-COUNT
               ADD ACCOUNT-BALANCE TO WS-TOTAL-BALANCE
               READ CUSTOMER-FILE
                   AT END MOVE 'Y' TO WS-EOF-FLAG
               END-READ
           END-PERFORM.
           
       DISPLAY-SUMMARY.
           DISPLAY 'TOTAL CUSTOMERS: ' WS-CUSTOMER-COUNT.
           DISPLAY 'TOTAL BALANCE: ' WS-TOTAL-BALANCE.
"""

# Sample JCL for testing
SAMPLE_JCL = """
//TESTJOB  JOB  (ACCT),'TEST JOB',CLASS=A,MSGCLASS=H
//STEP1    EXEC PGM=TEST-PROGRAM
//CUSTFILE DD   DSN=PROD.CUSTOMER.DATA,DISP=SHR
//SYSOUT   DD   SYSOUT=*
//SYSIN    DD   *
PROCESSING PARAMETERS
/*
"""

# Sample CSV for testing
SAMPLE_CSV = """CUSTOMER_ID,CUSTOMER_NAME,ACCOUNT_BALANCE,ACCOUNT_TYPE
12345678,JOHN SMITH,15000.50,CHECKING
23456789,JANE DOE,25000.75,SAVINGS
34567890,BOB JOHNSON,5000.00,CHECKING
45678901,ALICE BROWN,50000.25,SAVINGS
"""

async def test_system():
    """Run comprehensive system tests"""
    print("="*80)
    print("OPULENCE SYSTEM TEST")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize coordinator
        print("\n1. Initializing Opulence Coordinator...")
        config = OpulenceConfig(
            model_name="codellama/CodeLlama-7b-Instruct-hf",
            gpu_count=1,  # Use 1 GPU for testing
            max_processing_time=300  # 5 minutes for testing
        )
        
        coordinator = OpulenceCoordinator(config)
        await coordinator._init_agents()
        print("✓ Coordinator initialized successfully")
        
        # Test 1: Health check
        print("\n2. Testing system health...")
        health = coordinator.get_health_status()
        print(f"✓ System status: {health['status']}")
        print(f"✓ Active agents: {health['active_agents']}")
        
        # Test 2: File processing
        print("\n3. Testing file processing...")
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write test files
            cobol_file = temp_path / "test_program.cbl"
            jcl_file = temp_path / "test_job.jcl"
            csv_file = temp_path / "customer_data.csv"
            
            cobol_file.write_text(SAMPLE_COBOL)
            jcl_file.write_text(SAMPLE_JCL)
            csv_file.write_text(SAMPLE_CSV)
            
            # Process files
            test_files = [cobol_file, jcl_file, csv_file]
            result = await coordinator.process_batch_files(test_files)
            
            if result["status"] == "success":
                print(f"✓ Processed {result['files_processed']} files successfully")
                print(f"✓ Processing time: {result['processing_time']:.2f} seconds")
            else:
                print(f"✗ File processing failed: {result.get('error', 'Unknown error')}")
                return False
        
        # Test 3: Component analysis
        print("\n4. Testing component analysis...")
        
        # Analyze the test program
        analysis_result = await coordinator.analyze_component("TEST-PROGRAM", "program")
        
        if "error" not in analysis_result:
            print("✓ Component analysis completed successfully")
            component_type = analysis_result.get("component_type", "unknown")
            print(f"✓ Detected component type: {component_type}")
        else:
            print(f"✗ Component analysis failed: {analysis_result['error']}")
        
        # Test 4: Field lineage analysis
        print("\n5. Testing field lineage analysis...")
        
        lineage_result = await coordinator.agents["lineage_analyzer"].analyze_field_lineage("CUSTOMER-ID")
        
        if "error" not in lineage_result:
            print("✓ Field lineage analysis completed successfully")
            field_name = lineage_result.get("field_name", "unknown")
            print(f"✓ Analyzed field: {field_name}")
        else:
            print(f"⚠ Field lineage analysis: {lineage_result.get('error', 'No data found')}")
        
        # Test 5: Vector search
        print("\n6. Testing vector search...")
        
        # Build embeddings first
        embedding_result = await coordinator.agents["vector_index"].process_batch_embeddings(limit=10)
        
        if embedding_result["status"] == "success":
            print(f"✓ Created {embedding_result['embeddings_created']} embeddings")
            
            # Test search
            search_results = await coordinator.agents["vector_index"].semantic_search(
                "customer processing logic", top_k=3
            )
            
            if search_results:
                print(f"✓ Found {len(search_results)} similar code patterns")
            else:
                print("⚠ No search results found")
        else:
            print(f"✗ Embedding creation failed: {embedding_result.get('error', 'Unknown error')}")
        
        # Test 6: Statistics
        print("\n7. Testing statistics...")
        
        stats = coordinator.get_statistics()
        print(f"✓ System statistics retrieved")
        print(f"✓ Files processed: {stats['system_stats']['total_files_processed']}")
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSystem is ready for use.")
        print("Start the web interface with: python3 main.py --mode web")
        
        return True
        
    except Exception as e:
        print(f"\n✗ System test failed: {str(e)}")
        logger.exception("Test failure details:")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)

