#!/usr/bin/env python3
"""
Standalone Database Debug Script for Opulence
Usage: python debug_opulence_db.py [component_name]
"""

import sqlite3
import json
import sys
import argparse
from pathlib import Path

class OpulenceDBDebugger:
    def __init__(self, db_path="opulence_data.db"):
        self.db_path = db_path
        
    def check_database_exists(self):
        """Check if database file exists"""
        if not Path(self.db_path).exists():
            print(f"❌ Database file not found: {self.db_path}")
            return False
        
        file_size = Path(self.db_path).stat().st_size
        print(f"✅ Database file found: {self.db_path} ({file_size:,} bytes)")
        return True
    
    def get_database_overview(self):
        """Get overall database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n" + "="*50)
            print("DATABASE OVERVIEW")
            print("="*50)
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📋 Tables found: {tables}")
            
            # Get row counts for each table
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  📊 {table}: {count:,} rows")
                except Exception as e:
                    print(f"  ❌ {table}: Error counting rows - {e}")
            
            conn.close()
            return tables
            
        except Exception as e:
            print(f"❌ Database overview failed: {e}")
            return []
    
    def analyze_program_chunks(self):
        """Analyze program_chunks table in detail"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n" + "="*50)
            print("PROGRAM CHUNKS ANALYSIS")
            print("="*50)
            
            # Total chunks
            cursor.execute("SELECT COUNT(*) FROM program_chunks")
            total_chunks = cursor.fetchone()[0]
            
            if total_chunks == 0:
                print("❌ No chunks found in program_chunks table!")
                conn.close()
                return []
            
            print(f"📊 Total chunks: {total_chunks:,}")
            
            # Programs with chunk counts
            cursor.execute("""
                SELECT program_name, COUNT(*) as chunk_count 
                FROM program_chunks 
                GROUP BY program_name 
                ORDER BY chunk_count DESC 
                LIMIT 15
            """)
            programs = cursor.fetchall()
            
            print(f"\n📁 Programs found ({len(programs)} total):")
            for i, (prog, count) in enumerate(programs, 1):
                print(f"  {i:2d}. {prog:<30} ({count:,} chunks)")
            
            # Chunk types
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count 
                FROM program_chunks 
                GROUP BY chunk_type 
                ORDER BY count DESC
            """)
            chunk_types = cursor.fetchall()
            
            print(f"\n🏷️  Chunk types:")
            for chunk_type, count in chunk_types:
                print(f"  📌 {chunk_type:<25} ({count:,} chunks)")
            
            # Sample content from different chunk types
            print(f"\n📝 Sample content by chunk type:")
            for chunk_type, _ in chunk_types[:3]:  # Show first 3 types
                cursor.execute("""
                    SELECT program_name, chunk_id, content, metadata 
                    FROM program_chunks 
                    WHERE chunk_type = ? 
                    LIMIT 1
                """, (chunk_type,))
                
                sample = cursor.fetchone()
                if sample:
                    prog, chunk_id, content, metadata = sample
                    print(f"\n  🔸 {chunk_type} example:")
                    print(f"     Program: {prog}")
                    print(f"     Chunk ID: {chunk_id}")
                    print(f"     Content (first 150 chars): {content[:150]}...")
                    
                    if metadata:
                        try:
                            meta = json.loads(metadata)
                            print(f"     Metadata keys: {list(meta.keys())}")
                            
                            # Show some key metadata
                            for key in ['field_names', 'operations', 'main_purpose']:
                                if key in meta:
                                    value = meta[key]
                                    if isinstance(value, list):
                                        print(f"     {key}: {len(value)} items - {value[:3]}...")
                                    else:
                                        print(f"     {key}: {str(value)[:50]}...")
                        except:
                            print(f"     Metadata (raw, first 100 chars): {metadata[:100]}...")
            
            conn.close()
            return [prog for prog, _ in programs]
            
        except Exception as e:
            print(f"❌ Program chunks analysis failed: {e}")
            return []
    
    def search_component(self, component_name):
        """Search for a specific component in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n" + "="*50)
            print(f"COMPONENT SEARCH: '{component_name}'")
            print("="*50)
            
            # Search in program_chunks by program_name
            print("🔍 Searching by program name...")
            cursor.execute("""
                SELECT program_name, chunk_id, chunk_type, content, metadata
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                LIMIT 10
            """, (component_name, f"%{component_name}%"))
            
            program_matches = cursor.fetchall()
            
            if program_matches:
                print(f"✅ Found {len(program_matches)} chunks by program name:")
                for i, (prog, chunk_id, chunk_type, content, metadata) in enumerate(program_matches, 1):
                    print(f"  {i}. Program: {prog}")
                    print(f"     Chunk: {chunk_id} ({chunk_type})")
                    print(f"     Content: {content[:100]}...")
                    
                    if metadata:
                        try:
                            meta = json.loads(metadata)
                            print(f"     Metadata: {list(meta.keys())}")
                        except:
                            print(f"     Metadata: {metadata[:50]}...")
                    print()
            else:
                print("❌ No matches found by program name")
            
            # Search in content and metadata
            print("\n🔍 Searching in content and metadata...")
            cursor.execute("""
                SELECT program_name, chunk_id, chunk_type, content, metadata
                FROM program_chunks 
                WHERE content LIKE ? OR metadata LIKE ?
                LIMIT 10
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            content_matches = cursor.fetchall()
            
            if content_matches:
                print(f"✅ Found {len(content_matches)} chunks containing '{component_name}':")
                for i, (prog, chunk_id, chunk_type, content, metadata) in enumerate(content_matches, 1):
                    # Find where the component appears
                    content_lower = content.lower()
                    component_lower = component_name.lower()
                    
                    if component_lower in content_lower:
                        start = max(0, content_lower.find(component_lower) - 50)
                        end = min(len(content), start + 150)
                        snippet = content[start:end]
                        
                        print(f"  {i}. Program: {prog}")
                        print(f"     Chunk: {chunk_id} ({chunk_type})")
                        print(f"     Context: ...{snippet}...")
                        print()
            else:
                print("❌ No matches found in content")
            
            # Search in file_metadata if it exists
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata'")
                if cursor.fetchone():
                    print("\n🔍 Searching in file metadata...")
                    cursor.execute("""
                        SELECT file_name, table_name, fields, file_type
                        FROM file_metadata 
                        WHERE file_name LIKE ? OR table_name LIKE ? OR fields LIKE ?
                        LIMIT 5
                    """, (f"%{component_name}%", f"%{component_name}%", f"%{component_name}%"))
                    
                    file_matches = cursor.fetchall()
                    
                    if file_matches:
                        print(f"✅ Found {len(file_matches)} file metadata matches:")
                        for file_name, table_name, fields, file_type in file_matches:
                            print(f"  📄 File: {file_name}")
                            print(f"     Table: {table_name}")
                            print(f"     Type: {file_type}")
                            if fields:
                                print(f"     Fields: {fields[:100]}...")
                            print()
                    else:
                        print("❌ No matches found in file metadata")
            except:
                print("⚠️  file_metadata table not available")
            
            conn.close()
            
            # Summary
            total_matches = len(program_matches) + len(content_matches)
            print(f"\n📊 SEARCH SUMMARY:")
            print(f"   Program name matches: {len(program_matches)}")
            print(f"   Content matches: {len(content_matches)}")
            print(f"   Total matches: {total_matches}")
            
            if total_matches == 0:
                print(f"\n💡 SUGGESTIONS:")
                print(f"   1. Check spelling of '{component_name}'")
                print(f"   2. Try partial names (e.g., if searching for 'PROGRAM_NAME', try 'PROGRAM')")
                print(f"   3. Use the list above to find actual component names")
                print(f"   4. Check if files were properly processed and loaded")
            
            return total_matches > 0
            
        except Exception as e:
            print(f"❌ Component search failed: {e}")
            return False
    
    def suggest_test_components(self, programs_list):
        """Suggest components to test based on what's in the database"""
        if not programs_list:
            return
        
        print("\n" + "="*50)
        print("SUGGESTED TEST COMPONENTS")
        print("="*50)
        
        print("🧪 Try testing with these components:")
        
        # First 5 programs
        print("\n📁 Program names:")
        for i, prog in enumerate(programs_list[:5], 1):
            print(f"  {i}. {prog}")
        
        # Extract potential field names from a sample
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get some metadata with field names
            cursor.execute("""
                SELECT metadata FROM program_chunks 
                WHERE metadata IS NOT NULL AND metadata != ''
                LIMIT 10
            """)
            
            all_fields = set()
            for (metadata_str,) in cursor.fetchall():
                try:
                    metadata = json.loads(metadata_str)
                    if 'field_names' in metadata and isinstance(metadata['field_names'], list):
                        all_fields.update(metadata['field_names'][:5])  # Add first 5 fields
                    if len(all_fields) >= 10:
                        break
                except:
                    continue
            
            if all_fields:
                print("\n🏷️  Field names found in metadata:")
                for i, field in enumerate(sorted(list(all_fields))[:8], 1):
                    print(f"  {i}. {field}")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Could not extract field suggestions: {e}")
    
    def run_full_debug(self, component_name=None):
        """Run complete database debug"""
        print("🔍 OPULENCE DATABASE DEBUGGER")
        print("="*50)
        
        # Check database exists
        if not self.check_database_exists():
            return False
        
        # Get overview
        tables = self.get_database_overview()
        
        if not tables:
            print("❌ No tables found in database!")
            return False
        
        # Analyze program chunks
        programs = []
        if 'program_chunks' in tables:
            programs = self.analyze_program_chunks()
        else:
            print("❌ program_chunks table not found!")
        
        # Search for specific component if provided
        if component_name:
            found = self.search_component(component_name)
            if not found:
                print(f"\n❌ Component '{component_name}' not found!")
        else:
            # Suggest test components
            self.suggest_test_components(programs)
        
        print("\n" + "="*50)
        print("DEBUG COMPLETE")
        print("="*50)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Debug Opulence Database')
    parser.add_argument('component', nargs='?', help='Component name to search for')
    parser.add_argument('--db', default='opulence_data.db', help='Database path (default: opulence_data.db)')
    
    args = parser.parse_args()
    
    debugger = OpulenceDBDebugger(args.db)
    
    try:
        debugger.run_full_debug(args.component)
    except KeyboardInterrupt:
        print("\n\n⚠️  Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()