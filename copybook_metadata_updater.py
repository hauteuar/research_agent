# copybook_metadata_updater.py
"""
Copybook Metadata Updater - Streamlit App
Updates COBOL copybooks with metadata headers for Opulence analysis
"""

import streamlit as st
import os
from pathlib import Path
import shutil
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Copybook Metadata Updater",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_copybook_name_from_filename(filename):
    """Extract copybook name from filename"""
    return Path(filename).stem.upper()

def detect_existing_metadata(content):
    """Detect if copybook already has metadata header"""
    lines = content.split('\n')
    for i, line in enumerate(lines[:20):  # Check first 20 lines
        if '*================================================================' in line:
            # Look for end of metadata block
            for j in range(i+1, min(i+15, len(lines))):
                if '*================================================================' in lines[j]:
                    return True, i, j+1
    return False, 0, 0

def generate_metadata_header(copybook_name, filename, purpose, operations, layout_type):
    """Generate metadata header for copybook"""
    header = f"""*================================================================
* COPYBOOK: {copybook_name}
* PURPOSE: {purpose}
* FILE: {filename}
* OPERATIONS: {operations}
* LAYOUT_TYPE: {layout_type}
* UPDATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*================================================================"""
    return header

def update_copybook_with_metadata(content, copybook_name, filename, purpose, operations, layout_type):
    """Update copybook content with metadata header"""
    
    # Generate new metadata header
    new_header = generate_metadata_header(copybook_name, filename, purpose, operations, layout_type)
    
    # Check for existing metadata
    has_metadata, start_idx, end_idx = detect_existing_metadata(content)
    
    lines = content.split('\n')
    
    if has_metadata:
        # Replace existing metadata
        updated_lines = lines[:start_idx] + new_header.split('\n') + lines[end_idx:]
    else:
        # Add new metadata at the beginning
        # Skip any initial comment lines that might be copyright or system headers
        insert_idx = 0
        for i, line in enumerate(lines[:10]):
            if line.strip() and not line.strip().startswith('*'):
                insert_idx = i
                break
        
        updated_lines = lines[:insert_idx] + new_header.split('\n') + [''] + lines[insert_idx:]
    
    return '\n'.join(updated_lines)

def find_copybook_files(input_folder):
    """Find all copybook files in input folder"""
    copybook_extensions = ['.cpy', '.copy', '.cbl', '.cob']
    copybook_files = []
    
    input_path = Path(input_folder)
    if input_path.exists():
        for ext in copybook_extensions:
            copybook_files.extend(list(input_path.glob(f"**/*{ext}")))
    
    return copybook_files

def process_copybooks(input_folder, output_folder, file_configs):
    """Process copybooks with metadata updates"""
    results = {
        'processed': [],
        'errors': [],
        'skipped': []
    }
    
    # Ensure output folder exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file_path, config in file_configs.items():
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Update with metadata
            updated_content = update_copybook_with_metadata(
                content,
                config['copybook_name'],
                config['filename'],
                config['purpose'],
                config['operations'],
                config['layout_type']
            )
            
            # Write to output folder
            output_file = output_path / Path(file_path).name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            results['processed'].append({
                'original': str(file_path),
                'output': str(output_file),
                'copybook_name': config['copybook_name']
            })
            
        except Exception as e:
            results['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })
    
    return results

# Main Streamlit App
def main():
    st.markdown('<div class="main-header">📝 Copybook Metadata Updater</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>📋 Purpose:</strong> Add or update metadata headers in COBOL copybooks for better Opulence analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.header("📁 Input Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_folder = st.text_input(
            "Input Folder Path",
            placeholder="/path/to/copybooks",
            help="Folder containing your original copybooks"
        )
    
    with col2:
        output_folder = st.text_input(
            "Output Folder Path", 
            placeholder="/path/to/updated_copybooks",
            help="Folder where updated copybooks will be saved"
        )
    
    # Scan for copybooks
    if input_folder and Path(input_folder).exists():
        copybook_files = find_copybook_files(input_folder)
        
        if copybook_files:
            st.success(f"✅ Found {len(copybook_files)} copybook files")
            
            # File selection and configuration
            st.header("⚙️ Copybook Configuration")
            
            # Global settings
            st.subheader("🌐 Global Settings (Applied to All)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                global_filename = st.text_input(
                    "Associated File Name",
                    placeholder="TMS092THO",
                    help="Main file these copybooks relate to"
                )
            
            with col2:
                global_operations = st.selectbox(
                    "Default Operations",
                    ["READ/WRITE/UPDATE", "READ/WRITE", "READ", "WRITE", "UPDATE", "READ/UPDATE"],
                    help="Default operations for copybooks"
                )
            
            with col3:
                global_layout_type = st.selectbox(
                    "Default Layout Type",
                    ["UNIVERSAL", "INPUT_LAYOUT", "OUTPUT_LAYOUT", "UPDATE_LAYOUT", "WORKING_STORAGE"],
                    help="Default layout type"
                )
            
            # Individual file configuration
            st.subheader("📋 Individual File Configuration")
            
            file_configs = {}
            
            # Create tabs for easier management
            if len(copybook_files) > 5:
                # Use accordion for many files
                for i, file_path in enumerate(copybook_files):
                    with st.expander(f"📄 {file_path.name}", expanded=False):
                        file_configs[file_path] = configure_single_file(
                            file_path, global_filename, global_operations, global_layout_type, i
                        )
            else:
                # Show all files for small number
                for i, file_path in enumerate(copybook_files):
                    st.markdown(f"#### 📄 {file_path.name}")
                    file_configs[file_path] = configure_single_file(
                        file_path, global_filename, global_operations, global_layout_type, i
                    )
            
            # Process button
            st.header("🚀 Process Copybooks")
            
            if output_folder and st.button("🔄 Update All Copybooks", type="primary"):
                if not Path(output_folder).parent.exists():
                    st.error("❌ Output folder parent directory does not exist!")
                else:
                    with st.spinner("Processing copybooks..."):
                        results = process_copybooks(input_folder, output_folder, file_configs)
                    
                    # Display results
                    display_results(results)
        
        else:
            st.warning("⚠️ No copybook files found in the specified folder")
    
    elif input_folder:
        st.error("❌ Input folder does not exist")
    
    # Help section
    with st.expander("ℹ️ Help & Examples"):
        show_help_section()

def configure_single_file(file_path, global_filename, global_operations, global_layout_type, index):
    """Configure a single copybook file"""
    
    # Extract default copybook name
    default_copybook_name = extract_copybook_name_from_filename(file_path.name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        copybook_name = st.text_input(
            "Copybook Name",
            value=default_copybook_name,
            key=f"copybook_name_{index}",
            help="Name as referenced in COPY statements"
        )
        
        purpose = st.text_input(
            "Purpose Description",
            value=f"RECORD LAYOUT FOR {global_filename}",
            key=f"purpose_{index}",
            help="Brief description of copybook purpose"
        )
    
    with col2:
        operations = st.selectbox(
            "Operations",
            ["READ/WRITE/UPDATE", "READ/WRITE", "READ", "WRITE", "UPDATE", "READ/UPDATE"],
            index=["READ/WRITE/UPDATE", "READ/WRITE", "READ", "WRITE", "UPDATE", "READ/UPDATE"].index(global_operations),
            key=f"operations_{index}",
            help="Operations performed using this layout"
        )
        
        layout_type = st.selectbox(
            "Layout Type", 
            ["UNIVERSAL", "INPUT_LAYOUT", "OUTPUT_LAYOUT", "UPDATE_LAYOUT", "WORKING_STORAGE"],
            index=["UNIVERSAL", "INPUT_LAYOUT", "OUTPUT_LAYOUT", "UPDATE_LAYOUT", "WORKING_STORAGE"].index(global_layout_type),
            key=f"layout_type_{index}",
            help="Type of layout"
        )
    
    return {
        'copybook_name': copybook_name,
        'filename': global_filename,
        'purpose': purpose,
        'operations': operations,
        'layout_type': layout_type
    }

def display_results(results):
    """Display processing results"""
    
    if results['processed']:
        st.markdown(f"""
        <div class="success-box">
        <strong>✅ Successfully Processed: {len(results['processed'])} files</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Show processed files
        with st.expander("📋 Processed Files Details"):
            for item in results['processed']:
                st.write(f"**{item['copybook_name']}**: {Path(item['original']).name} → {Path(item['output']).name}")
    
    if results['errors']:
        st.markdown(f"""
        <div class="error-box">
        <strong>❌ Errors: {len(results['errors'])} files failed</strong>
        </div>
        """, unsafe_allow_html=True)
        
        for error in results['errors']:
            st.error(f"**{Path(error['file']).name}**: {error['error']}")
    
    if results['skipped']:
        st.warning(f"⚠️ Skipped: {len(results['skipped'])} files")

def show_help_section():
    """Show help and examples"""
    
    st.markdown("""
    ### 🎯 How to Use
    
    1. **Input Folder**: Specify folder containing your original copybooks
    2. **Output Folder**: Choose where updated copybooks will be saved
    3. **Configure**: Set metadata for each copybook
    4. **Process**: Click "Update All Copybooks" to generate updated files
    
    ### 📝 Metadata Fields
    
    - **Copybook Name**: Name used in COPY statements (usually filename without extension)
    - **Associated File**: Main file these copybooks relate to (e.g., TMS092THO)
    - **Purpose**: Brief description of what the copybook defines
    - **Operations**: What operations use this layout (READ/WRITE/UPDATE)
    - **Layout Type**: Type of layout (UNIVERSAL, INPUT_LAYOUT, etc.)
    
    ### 🔧 Example Generated Metadata
    
    ```cobol
    *================================================================
    * COPYBOOK: TMSCOTHI
    * PURPOSE: INPUT LAYOUT FOR TMS092THO
    * FILE: TMS092THO
    * OPERATIONS: READ/WRITE
    * LAYOUT_TYPE: INPUT_LAYOUT
    * UPDATED: 2024-01-15 14:30:25
    *================================================================
    ```
    
    ### 📁 File Extensions Supported
    
    - `.cpy` - Standard copybook extension
    - `.copy` - Alternative copybook extension  
    - `.cbl` - COBOL source files
    - `.cob` - COBOL source files
    
    ### ⚠️ Important Notes
    
    - Original files are **NOT modified** - only copies are created in output folder
    - If metadata already exists, it will be **replaced** with new metadata
    - Backup your original files before processing
    - Use meaningful file and folder names for better organization
    """)

if __name__ == "__main__":
    main()