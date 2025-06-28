# Debug script to identify component analysis issues

import sqlite3
import json
from pathlib import Path

def debug_component_analysis(component_name, db_path="opulence_data.db"):
    """Debug why component analysis is failing"""
    
    print(f"🔍 Debugging component analysis for: {component_name}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Check if database tables exist
        print("1. Checking database tables...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['program_chunks', 'file_metadata', 'field_lineage', 'lineage_nodes', 'lineage_edges']
        for table in required_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   ✅ {table}: {count} records")
            else:
                print(f"   ❌ {table}: TABLE MISSING")
        
        print()
        
        # 2. Check program_chunks for the component
        print("2. Checking program_chunks...")
        cursor.execute("SELECT DISTINCT program_name FROM program_chunks")
        all_programs = [row[0] for row in cursor.fetchall()]
        
        print(f"   Available programs ({len(all_programs)}):")
        for prog in all_programs[:10]:  # Show first 10
            print(f"     - {prog}")
        if len(all_programs) > 10:
            print(f"     ... and {len(all_programs) - 10} more")
        
        # Check exact and partial matches
        exact_match = component_name in all_programs
        partial_matches = [p for p in all_programs if component_name.lower() in p.lower()]
        
        print(f"\n   🎯 Exact match for '{component_name}': {exact_match}")
        if partial_matches:
            print(f"   🔍 Partial matches:")
            for match in partial_matches:
                print(f"     - {match}")
        
        if exact_match or partial_matches:
            # Get chunks for the component
            search_name = component_name if exact_match else partial_matches[0]
            cursor.execute("""
                SELECT chunk_id, chunk_type, LENGTH(content) as content_length, metadata
                FROM program_chunks 
                WHERE program_name = ?
                ORDER BY chunk_id
            """, (search_name,))
            
            chunks = cursor.fetchall()
            print(f"\n   📦 Chunks for '{search_name}': {len(chunks)}")
            
            for chunk_id, chunk_type, content_len, metadata_str in chunks:
                metadata = json.loads(metadata_str) if metadata_str else {}
                print(f"     - {chunk_id} ({chunk_type}): {content_len} chars")
                if metadata:
                    print(f"       Metadata keys: {list(metadata.keys())}")
        
        print()
        
        # 3. Check field_lineage table
        print("3. Checking field_lineage...")
        cursor.execute("SELECT COUNT(*) FROM field_lineage")
        lineage_count = cursor.fetchone()[0]
        print(f"   📊 Total field lineage records: {lineage_count}")
        
        if lineage_count > 0:
            cursor.execute("SELECT DISTINCT program_name FROM field_lineage LIMIT 10")
            lineage_programs = [row[0] for row in cursor.fetchall()]
            print(f"   Programs in lineage: {lineage_programs}")
        
        print()
        
        # 4. Check lineage tables
        print("4. Checking lineage tables...")
        for table in ['lineage_nodes', 'lineage_edges']:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count} records")
                
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    samples = cursor.fetchall()
                    print(f"     Sample data: {len(samples)} rows")
        
        print()
        
        # 5. Check vector embeddings
        print("5. Checking vector embeddings...")
        if 'vector_embeddings' in tables:
            cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
            embed_count = cursor.fetchone()[0]
            print(f"   📈 Vector embeddings: {embed_count}")
            
            if embed_count > 0:
                cursor.execute("""
                    SELECT pc.program_name, COUNT(*) as embed_count
                    FROM vector_embeddings ve
                    JOIN program_chunks pc ON ve.chunk_id = pc.id
                    GROUP BY pc.program_name
                    ORDER BY embed_count DESC
                    LIMIT 5
                """)
                top_embedded = cursor.fetchall()
                print(f"   Top embedded programs:")
                for prog, count in top_embedded:
                    print(f"     - {prog}: {count} embeddings")
        else:
            print("   ❌ vector_embeddings table missing")
        
        conn.close()
        
        # 6. Check file system
        print("\n6. Checking file system...")
        faiss_index = Path("opulence_faiss.index")
        chroma_db = Path("./chroma_db")
        
        print(f"   FAISS index exists: {faiss_index.exists()}")
        if faiss_index.exists():
            print(f"   FAISS index size: {faiss_index.stat().st_size / 1024:.1f} KB")
        
        print(f"   ChromaDB exists: {chroma_db.exists()}")
        if chroma_db.exists():
            print(f"   ChromaDB contents: {list(chroma_db.iterdir())}")
        
        print("\n" + "=" * 60)
        
        # 7. Provide recommendations
        print("🔧 RECOMMENDATIONS:")
        
        if not exact_match and not partial_matches:
            print("❌ Component not found in database")
            print("   - Check if files were properly processed")
            print("   - Verify component name spelling")
            print("   - Try using exact program names from the list above")
        
        if lineage_count == 0:
            print("❌ No field lineage data")
            print("   - Lineage analysis requires field tracking during parsing")
            print("   - Re-process files to generate lineage data")
        
        if 'vector_embeddings' not in tables or embed_count == 0:
            print("❌ No vector embeddings")
            print("   - Run vector indexing after file processing")
            print("   - Check if VectorIndexAgent is working properly")
        
        return {
            "component_found": exact_match or bool(partial_matches),
            "chunks_available": len(chunks) if (exact_match or partial_matches) else 0,
            "lineage_available": lineage_count > 0,
            "embeddings_available": embed_count > 0 if 'vector_embeddings' in tables else False,
            "recommendations": []
        }
        
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        return {"error": str(e)}

def check_agent_initialization():
    """Check if agents can be properly initialized"""
    print("\n🤖 Checking agent initialization...")
    
    try:
        # Try to create agents without coordinator
        from agents.lineage_analyzer_agent import LineageAnalyzerAgent
        from agents.logic_analyzer_agent import LogicAnalyzerAgent
        from agents.vector_index_agent import VectorIndexAgent
        
        print("✅ Agent imports successful")
        
        # Try basic initialization
        lineage_agent = LineageAnalyzerAgent()
        logic_agent = LogicAnalyzerAgent() 
        vector_agent = VectorIndexAgent()
        
        print("✅ Agent creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {str(e)}")
        return False

def fix_component_analysis_issues():
    """Provide fixes for common component analysis issues"""
    
    fixes = """
🔧 FIXES FOR COMPONENT ANALYSIS ISSUES:

1. **Database Initialization Fix:**
   - Ensure all required tables exist
   - Run the coordinator initialization properly
   
2. **LLM Engine Fix:**
   - Initialize engines before using agents
   - Handle GPU allocation properly
   
3. **Lineage Data Fix:**
   - Re-process files to generate proper metadata
   - Ensure field extraction is working
   
4. **Component Lookup Fix:**
   - Use exact program names from database
   - Handle case sensitivity
   
5. **Vector Index Fix:**
   - Run vector indexing after file processing
   - Check if ChromaDB is properly initialized

6. **Agent Communication Fix:**
   - Ensure coordinator is properly passing engines to agents
   - Check if agents can access shared resources
"""
    
    print(fixes)

# Quick usage functions
def quick_debug(component_name):
    """Quick debug for a component"""
    result = debug_component_analysis(component_name)
    check_agent_initialization()
    fix_component_analysis_issues()
    return result

# Example usage:
if __name__ == "__main__":
    # Replace with your component name
    component_name = "YOUR_COMPONENT_NAME"
    result = quick_debug(component_name)
    print(f"\nDebug result: {result}")

"""
Clear all indices and rebuild:
python# Remove FAISS index
os.remove("opulence_faiss.index")

# Clear ChromaDB
import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)

# Rebuild from scratch

Reset database:
python# Backup first
shutil.copy("opulence_data.db", "opulence_data_backup.db")

# Clear problematic tables
conn = sqlite3.connect("opulence_data.db")
conn.execute("DELETE FROM vector_embeddings")
conn.commit() """