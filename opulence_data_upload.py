"""
Complete Data Upload & Ingestion Module for Mainframe Analysis System
Handles all types of data uploads: source files, data files, configuration files
"""

import os
import shutil
import zipfile
import tarfile
import ftplib
import paramiko
import boto3
import pandas as pd
import sqlite3
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from datetime import datetime
import hashlib
import mimetypes
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

@dataclass
class UploadResult:
    """Result of file upload operation"""
    success: bool
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    detected_type: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataSource:
    """Configuration for different data sources"""
    source_type: str  # 'local', 'ftp', 'sftp', 's3', 'http', 'database'
    connection_config: Dict[str, Any] = field(default_factory=dict)
    file_patterns: List[str] = field(default_factory=list)
    target_directory: str = "uploads"

class UniversalDataUploader:
    """Universal data uploader supporting multiple sources and file types"""
    
    def __init__(self, base_upload_dir: str = "uploads"):
        self.base_upload_dir = Path(base_upload_dir)
        self.setup_directories()
        
        # Supported file types and their categories
        self.file_type_mapping = {
            # Mainframe source files
            '.cbl': 'COBOL_PROGRAM',
            '.cob': 'COBOL_PROGRAM', 
            '.cobol': 'COBOL_PROGRAM',
            '.jcl': 'JCL_JOB',
            '.job': 'JCL_JOB',
            '.proc': 'JCL_PROC',
            '.cpy': 'COPYBOOK',
            '.copy': 'COPYBOOK',
            '.sql': 'DB2_SQL',
            '.ddl': 'DB2_DDL',
            '.map': 'CICS_MAP',
            '.bms': 'CICS_BMS',
            
            # Data files
            '.csv': 'DATA_CSV',
            '.txt': 'DATA_TEXT',
            '.dat': 'DATA_BINARY',
            '.vsam': 'DATA_VSAM',
            '.seq': 'DATA_SEQUENTIAL',
            '.xlsx': 'DATA_EXCEL',
            '.json': 'DATA_JSON',
            '.xml': 'DATA_XML',
            
            # Documentation
            '.pdf': 'DOCUMENTATION',
            '.doc': 'DOCUMENTATION',
            '.docx': 'DOCUMENTATION',
            '.md': 'DOCUMENTATION',
            
            # Configuration
            '.yaml': 'CONFIG',
            '.yml': 'CONFIG',
            '.properties': 'CONFIG',
            '.cfg': 'CONFIG',
            
            # Archives
            '.zip': 'ARCHIVE',
            '.tar': 'ARCHIVE',
            '.gz': 'ARCHIVE',
            '.7z': 'ARCHIVE'
        }
        
        self.upload_stats = {
            'total_uploaded': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_size': 0
        }
    
    def setup_directories(self):
        """Setup directory structure for uploads"""
        directories = [
            'source_files/cobol',
            'source_files/jcl', 
            'source_files/copybooks',
            'source_files/sql',
            'source_files/cics',
            'data_files/csv',
            'data_files/sequential',
            'data_files/vsam',
            'data_files/excel',
            'documentation',
            'configuration',
            'archives',
            'temp',
            'processed',
            'failed'
        ]
        
        for directory in directories:
            (self.base_upload_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def upload_from_streamlit(self, uploaded_files: List[Any]) -> List[UploadResult]:
        """Handle file uploads from Streamlit interface"""
        results = []
        
        for uploaded_file in uploaded_files:
            try:
                # Determine file type
                file_extension = Path(uploaded_file.name).suffix.lower()
                detected_type = self.file_type_mapping.get(file_extension, 'UNKNOWN')
                
                # Determine target directory
                target_dir = self._get_target_directory(detected_type)
                target_path = target_dir / uploaded_file.name
                
                # Save file
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Calculate file info
                file_size = target_path.stat().st_size
                
                # Create result
                result = UploadResult(
                    success=True,
                    file_path=str(target_path),
                    file_name=uploaded_file.name,
                    file_size=file_size,
                    file_type=file_extension,
                    detected_type=detected_type,
                    metadata={
                        'upload_time': datetime.now().isoformat(),
                        'upload_method': 'streamlit',
                        'mime_type': uploaded_file.type if hasattr(uploaded_file, 'type') else None
                    }
                )
                
                # Handle special file types
                if detected_type == 'ARCHIVE':
                    extracted_files = self._extract_archive(target_path)
                    result.metadata['extracted_files'] = extracted_files
                
                elif detected_type in ['DATA_CSV', 'DATA_EXCEL']:
                    data_info = self._analyze_data_file(target_path)
                    result.metadata['data_info'] = data_info
                
                results.append(result)
                self.upload_stats['successful_uploads'] += 1
                self.upload_stats['total_size'] += file_size
                
            except Exception as e:
                result = UploadResult(
                    success=False,
                    file_path="",
                    file_name=uploaded_file.name,
                    file_size=0,
                    file_type=file_extension,
                    detected_type='ERROR',
                    error_message=str(e)
                )
                results.append(result)
                self.upload_stats['failed_uploads'] += 1
        
        self.upload_stats['total_uploaded'] += len(uploaded_files)
        return results
    
    def upload_from_local_directory(self, directory_path: str, 
                                  recursive: bool = True,
                                  file_patterns: List[str] = None) -> List[UploadResult]:
        """Upload files from local directory"""
        results = []
        directory = Path(directory_path)
        
        if not directory.exists():
            return [UploadResult(
                success=False,
                file_path="",
                file_name=directory_path,
                file_size=0,
                file_type="",
                detected_type='ERROR',
                error_message="Directory does not exist"
            )]
        
        # Get file patterns or use default
        if file_patterns is None:
            file_patterns = ['*.*']
        
        # Find files
        files_to_upload = []
        for pattern in file_patterns:
            if recursive:
                files_to_upload.extend(directory.rglob(pattern))
            else:
                files_to_upload.extend(directory.glob(pattern))
        
        # Upload files
        for file_path in files_to_upload:
            if file_path.is_file():
                result = self._upload_single_file(file_path)
                results.append(result)
        
        return results
    
    def upload_from_ftp(self, data_source: DataSource) -> List[UploadResult]:
        """Upload files from FTP server"""
        results = []
        
        try:
            config = data_source.connection_config
            ftp = ftplib.FTP()
            ftp.connect(config['host'], config.get('port', 21))
            ftp.login(config['username'], config['password'])
            
            # Change to remote directory if specified
            if 'remote_directory' in config:
                ftp.cwd(config['remote_directory'])
            
            # List files
            file_list = ftp.nlst()
            
            # Download matching files
            for file_name in file_list:
                # Check if file matches patterns
                if self._matches_patterns(file_name, data_source.file_patterns):
                    local_path = self.base_upload_dir / 'temp' / file_name
                    
                    try:
                        with open(local_path, 'wb') as local_file:
                            ftp.retrbinary(f'RETR {file_name}', local_file.write)
                        
                        # Process the downloaded file
                        result = self._process_downloaded_file(local_path, 'ftp')
                        results.append(result)
                        
                    except Exception as e:
                        results.append(UploadResult(
                            success=False,
                            file_path="",
                            file_name=file_name,
                            file_size=0,
                            file_type="",
                            detected_type='ERROR',
                            error_message=f"FTP download error: {e}"
                        ))
            
            ftp.quit()
            
        except Exception as e:
            results.append(UploadResult(
                success=False,
                file_path="",
                file_name="FTP_CONNECTION",
                file_size=0,
                file_type="",
                detected_type='ERROR',
                error_message=f"FTP connection error: {e}"
            ))
        
        return results
    
    def upload_from_sftp(self, data_source: DataSource) -> List[UploadResult]:
        """Upload files from SFTP server"""
        results = []
        
        try:
            config = data_source.connection_config
            
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            ssh.connect(
                hostname=config['host'],
                port=config.get('port', 22),
                username=config['username'],
                password=config.get('password'),
                key_filename=config.get('private_key_path')
            )
            
            # Create SFTP client
            sftp = ssh.open_sftp()
            
            # List files in remote directory
            remote_dir = config.get('remote_directory', '.')
            file_list = sftp.listdir(remote_dir)
            
            # Download matching files
            for file_name in file_list:
                if self._matches_patterns(file_name, data_source.file_patterns):
                    remote_path = f"{remote_dir}/{file_name}"
                    local_path = self.base_upload_dir / 'temp' / file_name
                    
                    try:
                        sftp.get(remote_path, local_path)
                        
                        # Process the downloaded file
                        result = self._process_downloaded_file(local_path, 'sftp')
                        results.append(result)
                        
                    except Exception as e:
                        results.append(UploadResult(
                            success=False,
                            file_path="",
                            file_name=file_name,
                            file_size=0,
                            file_type="",
                            detected_type='ERROR',
                            error_message=f"SFTP download error: {e}"
                        ))
            
            sftp.close()
            ssh.close()
            
        except Exception as e:
            results.append(UploadResult(
                success=False,
                file_path="",
                file_name="SFTP_CONNECTION",
                file_size=0,
                file_type="",
                detected_type='ERROR',
                error_message=f"SFTP connection error: {e}"
            ))
        
        return results
    
    def upload_from_s3(self, data_source: DataSource) -> List[UploadResult]:
        """Upload files from AWS S3"""
        results = []
        
        try:
            config = data_source.connection_config
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=config['aws_access_key_id'],
                aws_secret_access_key=config['aws_secret_access_key'],
                region_name=config.get('region', 'us-east-1')
            )
            
            bucket_name = config['bucket_name']
            prefix = config.get('prefix', '')
            
            # List objects
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_name = obj['Key']
                    
                    if self._matches_patterns(file_name, data_source.file_patterns):
                        local_path = self.base_upload_dir / 'temp' / Path(file_name).name
                        
                        try:
                            # Download file
                            s3_client.download_file(bucket_name, file_name, local_path)
                            
                            # Process the downloaded file
                            result = self._process_downloaded_file(local_path, 's3')
                            result.metadata['s3_key'] = file_name
                            result.metadata['s3_bucket'] = bucket_name
                            results.append(result)
                            
                        except Exception as e:
                            results.append(UploadResult(
                                success=False,
                                file_path="",
                                file_name=file_name,
                                file_size=0,
                                file_type="",
                                detected_type='ERROR',
                                error_message=f"S3 download error: {e}"
                            ))
            
        except Exception as e:
            results.append(UploadResult(
                success=False,
                file_path="",
                file_name="S3_CONNECTION",
                file_size=0,
                file_type="",
                detected_type='ERROR',
                error_message=f"S3 connection error: {e}"
            ))
        
        return results
    
    def upload_from_http_url(self, urls: List[str]) -> List[UploadResult]:
        """Download files from HTTP URLs"""
        results = []
        
        for url in urls:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Extract filename from URL or Content-Disposition
                file_name = self._extract_filename_from_url(url, response.headers)
                local_path = self.base_upload_dir / 'temp' / file_name
                
                # Download file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Process the downloaded file
                result = self._process_downloaded_file(local_path, 'http')
                result.metadata['source_url'] = url
                results.append(result)
                
            except Exception as e:
                results.append(UploadResult(
                    success=False,
                    file_path="",
                    file_name=url,
                    file_size=0,
                    file_type="",
                    detected_type='ERROR',
                    error_message=f"HTTP download error: {e}"
                ))
        
        return results
    
    def upload_database_export(self, data_source: DataSource) -> List[UploadResult]:
        """Export data from database and upload as files"""
        results = []
        
        try:
            config = data_source.connection_config
            
            # Create database connection (example for DB2)
            if config['db_type'].lower() == 'db2':
                import ibm_db
                conn_str = f"""
                DATABASE={config['database']};
                HOSTNAME={config['hostname']};
                PORT={config['port']};
                PROTOCOL=TCPIP;
                UID={config['username']};
                PWD={config['password']};
                """
                conn = ibm_db.connect(conn_str, "", "")
                
                # Export specified tables
                for table_name in config.get('tables', []):
                    try:
                        # Query data
                        sql = f"SELECT * FROM {table_name}"
                        if 'row_limit' in config:
                            sql += f" FETCH FIRST {config['row_limit']} ROWS ONLY"
                        
                        df = pd.read_sql(sql, conn)
                        
                        # Save as CSV
                        csv_filename = f"{table_name}_export.csv"
                        csv_path = self.base_upload_dir / 'data_files' / 'csv' / csv_filename
                        df.to_csv(csv_path, index=False)
                        
                        # Create result
                        result = UploadResult(
                            success=True,
                            file_path=str(csv_path),
                            file_name=csv_filename,
                            file_size=csv_path.stat().st_size,
                            file_type='.csv',
                            detected_type='DATA_CSV',
                            metadata={
                                'source_table': table_name,
                                'export_time': datetime.now().isoformat(),
                                'record_count': len(df),
                                'upload_method': 'database_export'
                            }
                        )
                        results.append(result)
                        
                    except Exception as e:
                        results.append(UploadResult(
                            success=False,
                            file_path="",
                            file_name=table_name,
                            file_size=0,
                            file_type=".csv",
                            detected_type='ERROR',
                            error_message=f"Table export error: {e}"
                        ))
                
                ibm_db.close(conn)
            
        except Exception as e:
            results.append(UploadResult(
                success=False,
                file_path="",
                file_name="DATABASE_CONNECTION",
                file_size=0,
                file_type="",
                detected_type='ERROR',
                error_message=f"Database connection error: {e}"
            ))
        
        return results
    
    def _upload_single_file(self, file_path: Path) -> UploadResult:
        """Upload a single file"""
        try:
            # Determine file type
            file_extension = file_path.suffix.lower()
            detected_type = self.file_type_mapping.get(file_extension, 'UNKNOWN')
            
            # Determine target directory
            target_dir = self._get_target_directory(detected_type)
            target_path = target_dir / file_path.name
            
            # Copy file
            shutil.copy2(file_path, target_path)
            
            # Calculate file info
            file_size = target_path.stat().st_size
            
            # Create result
            result = UploadResult(
                success=True,
                file_path=str(target_path),
                file_name=file_path.name,
                file_size=file_size,
                file_type=file_extension,
                detected_type=detected_type,
                metadata={
                    'upload_time': datetime.now().isoformat(),
                    'upload_method': 'local_directory',
                    'source_path': str(file_path)
                }
            )
            
            # Handle special file types
            if detected_type == 'ARCHIVE':
                extracted_files = self._extract_archive(target_path)
                result.metadata['extracted_files'] = extracted_files
            
            elif detected_type in ['DATA_CSV', 'DATA_EXCEL']:
                data_info = self._analyze_data_file(target_path)
                result.metadata['data_info'] = data_info
            
            self.upload_stats['successful_uploads'] += 1
            self.upload_stats['total_size'] += file_size
            
            return result
            
        except Exception as e:
            self.upload_stats['failed_uploads'] += 1
            return UploadResult(
                success=False,
                file_path="",
                file_name=file_path.name,
                file_size=0,
                file_type=file_extension,
                detected_type='ERROR',
                error_message=str(e)
            )
    
    def _get_target_directory(self, detected_type: str) -> Path:
        """Get target directory based on detected file type"""
        type_mapping = {
            'COBOL_PROGRAM': 'source_files/cobol',
            'JCL_JOB': 'source_files/jcl',
            'JCL_PROC': 'source_files/jcl',
            'COPYBOOK': 'source_files/copybooks',
            'DB2_SQL': 'source_files/sql',
            'DB2_DDL': 'source_files/sql',
            'CICS_MAP': 'source_files/cics',
            'CICS_BMS': 'source_files/cics',
            'DATA_CSV': 'data_files/csv',
            'DATA_TEXT': 'data_files/sequential',
            'DATA_BINARY': 'data_files/sequential',
            'DATA_VSAM': 'data_files/vsam',
            'DATA_SEQUENTIAL': 'data_files/sequential',
            'DATA_EXCEL': 'data_files/excel',
            'DATA_JSON': 'data_files/csv',  # Treat JSON as structured data
            'DATA_XML': 'data_files/csv',   # Treat XML as structured data
            'DOCUMENTATION': 'documentation',
            'CONFIG': 'configuration',
            'ARCHIVE': 'archives'
        }
        
        subdir = type_mapping.get(detected_type, 'temp')
        return self.base_upload_dir / subdir
    
    def _extract_archive(self, archive_path: Path) -> List[str]:
        """Extract archive files and return list of extracted files"""
        extracted_files = []
        extract_dir = self.base_upload_dir / 'temp' / f"extracted_{archive_path.stem}"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    extracted_files = zip_ref.namelist()
            
            elif archive_path.suffix.lower() in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    extracted_files = tar_ref.getnames()
            
            # Process extracted files
            for extracted_file in extracted_files:
                file_path = extract_dir / extracted_file
                if file_path.is_file():
                    self._upload_single_file(file_path)
            
        except Exception as e:
            print(f"Error extracting archive {archive_path}: {e}")
        
        return extracted_files
    
    def _analyze_data_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze data file and return metadata"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=100)  # Sample first 100 rows
                return {
                    'columns': list(df.columns),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'sample_data': df.head(3).to_dict('records')
                }
            
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=100)
                return {
                    'columns': list(df.columns),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'sample_data': df.head(3).to_dict('records')
                }
            
        except Exception as e:
            return {'error': f"Error analyzing data file: {e}"}
        
        return {}
    
    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any of the patterns"""
        if not patterns:
            return True
        
        import fnmatch
        return any(fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in patterns)
    
    def _process_downloaded_file(self, temp_path: Path, source_type: str) -> UploadResult:
        """Process a downloaded file and move to appropriate directory"""
        # Determine file type and move to appropriate directory
        result = self._upload_single_file(temp_path)
        result.metadata['upload_method'] = source_type
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
        
        return result
    
    def _extract_filename_from_url(self, url: str, headers: Dict[str, str]) -> str:
        """Extract filename from URL or HTTP headers"""
        # Try Content-Disposition header first
        content_disposition = headers.get('content-disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
            return filename
        
        # Extract from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        
        if not filename:
            filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return filename
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Get upload statistics"""
        return {
            **self.upload_stats,
            'success_rate': (self.upload_stats['successful_uploads'] / 
                           max(self.upload_stats['total_uploaded'], 1)) * 100,
            'total_size_mb': self.upload_stats['total_size'] / (1024 * 1024)
        }
    
    def list_uploaded_files(self, file_type: str = None) -> List[Dict[str, Any]]:
        """List all uploaded files with optional filtering by type"""
        files = []
        
        # Search through all subdirectories
        for root, dirs, file_names in os.walk(self.base_upload_dir):
            for file_name in file_names:
                file_path = Path(root) / file_name
                
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    detected_type = self.file_type_mapping.get(file_extension, 'UNKNOWN')
                    
                    if file_type is None or detected_type == file_type:
                        stat = file_path.stat()
                        files.append({
                            'name': file_name,
                            'path': str(file_path),
                            'type': detected_type,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'extension': file_extension
                        })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)

# Streamlit UI for Data Upload
def create_data_upload_ui():
    """Streamlit UI for comprehensive data upload"""
    st.title("📤 Universal Data Upload System")
    st.markdown("### Upload data from multiple sources for mainframe analysis")
    
    # Initialize uploader
    if 'data_uploader' not in st.session_state:
        st.session_state.data_uploader = UniversalDataUploader()
    
    uploader = st.session_state.data_uploader
    
    # Upload method selection
    st.subheader("📂 Select Upload Method")
    upload_method = st.selectbox(
        "Choose how to upload your data:",
        [
            "Streamlit File Upload",
            "Local Directory",
            "FTP Server", 
            "SFTP Server",
            "AWS S3",
            "HTTP URLs",
            "Database Export"
        ]
    )
    
    upload_results = []
    
    # Streamlit File Upload
    if upload_method == "Streamlit File Upload":
        st.subheader("📁 Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mainframe Source Files:**")
            source_files = st.file_uploader(
                "Upload COBOL, JCL, Copybooks, SQL files",
                accept_multiple_files=True,
                type=['cbl', 'cob', 'cobol', 'jcl', 'job', 'cpy', 'copy', 'sql', 'txt'],
                key="source_files"
            )
        
        with col2:
            st.write("**Data Files:**")
            data_files = st.file_uploader(
                "Upload CSV, Excel, Sequential data files",
                accept_multiple_files=True,
                type=['csv', 'xlsx', 'xls', 'txt', 'dat'],
                key="data_files"
            )
        
        # Additional file types
        st.write("**Other Files:**")
        other_files = st.file_uploader(
            "Upload documentation, configuration, archive files",
            accept_multiple_files=True,
            type=['pdf', 'doc', 'docx', 'yaml', 'yml', 'json', 'xml', 'zip', 'tar', 'gz'],
            key="other_files"
        )
        
        if st.button("🚀 Upload Files"):
            all_files = (source_files or []) + (data_files or []) + (other_files or [])
            
            if all_files:
                with st.spinner("Uploading files..."):
                    upload_results = uploader.upload_from_streamlit(all_files)
            else:
                st.warning("Please select files to upload")
    
    # Local Directory Upload
    elif upload_method == "Local Directory":
        st.subheader("📂 Local Directory Upload")
        
        directory_path = st.text_input("Directory Path:", placeholder="/path/to/mainframe/files")
        recursive = st.checkbox("Include subdirectories", value=True)
        
        st.write("**File Patterns (optional):**")
        patterns_text = st.text_area(
            "Enter file patterns (one per line):",
            placeholder="*.cbl\n*.jcl\n*.csv\n*.dat",
            height=100
        )
        
        if st.button("📂 Upload from Directory"):
            if directory_path:
                patterns = [p.strip() for p in patterns_text.split('\n') if p.strip()] if patterns_text else None
                
                with st.spinner("Scanning directory and uploading files..."):
                    upload_results = uploader.upload_from_local_directory(
                        directory_path, recursive, patterns
                    )
            else:
                st.error("Please enter a directory path")
    
    # FTP Upload
    elif upload_method == "FTP Server":
        st.subheader("🌐 FTP Server Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ftp_host = st.text_input("FTP Host:")
            ftp_port = st.number_input("FTP Port:", value=21, min_value=1, max_value=65535)
            ftp_username = st.text_input("Username:")
        
        with col2:
            ftp_password = st.text_input("Password:", type="password")
            ftp_directory = st.text_input("Remote Directory:", placeholder="/mainframe/data")
            ftp_patterns = st.text_input("File Patterns:", placeholder="*.cbl,*.jcl,*.csv")
        
        if st.button("📥 Download from FTP"):
            if all([ftp_host, ftp_username, ftp_password]):
                patterns = [p.strip() for p in ftp_patterns.split(',') if p.strip()] if ftp_patterns else ['*']
                
                data_source = DataSource(
                    source_type='ftp',
                    connection_config={
                        'host': ftp_host,
                        'port': ftp_port,
                        'username': ftp_username,
                        'password': ftp_password,
                        'remote_directory': ftp_directory
                    },
                    file_patterns=patterns
                )
                
                with st.spinner("Connecting to FTP and downloading files..."):
                    upload_results = uploader.upload_from_ftp(data_source)
            else:
                st.error("Please fill in all required FTP connection details")
    
    # SFTP Upload
    elif upload_method == "SFTP Server":
        st.subheader("🔐 SFTP Server Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sftp_host = st.text_input("SFTP Host:")
            sftp_port = st.number_input("SFTP Port:", value=22, min_value=1, max_value=65535)
            sftp_username = st.text_input("Username:")
            sftp_password = st.text_input("Password:", type="password")
        
        with col2:
            sftp_directory = st.text_input("Remote Directory:", placeholder="/home/user/mainframe")
            sftp_key_file = st.text_input("Private Key File (optional):", placeholder="/path/to/private_key")
            sftp_patterns = st.text_input("File Patterns:", placeholder="*.cbl,*.jcl,*.csv")
        
        if st.button("📥 Download from SFTP"):
            if all([sftp_host, sftp_username]) and (sftp_password or sftp_key_file):
                patterns = [p.strip() for p in sftp_patterns.split(',') if p.strip()] if sftp_patterns else ['*']
                
                config = {
                    'host': sftp_host,
                    'port': sftp_port,
                    'username': sftp_username,
                    'remote_directory': sftp_directory
                }
                
                if sftp_password:
                    config['password'] = sftp_password
                if sftp_key_file:
                    config['private_key_path'] = sftp_key_file
                
                data_source = DataSource(
                    source_type='sftp',
                    connection_config=config,
                    file_patterns=patterns
                )
                
                with st.spinner("Connecting to SFTP and downloading files..."):
                    upload_results = uploader.upload_from_sftp(data_source)
            else:
                st.error("Please provide host, username, and either password or private key")
    
    # AWS S3 Upload
    elif upload_method == "AWS S3":
        st.subheader("☁️ AWS S3 Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            aws_access_key = st.text_input("AWS Access Key ID:")
            aws_secret_key = st.text_input("AWS Secret Access Key:", type="password")
            s3_region = st.selectbox("AWS Region:", [
                "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"
            ])
        
        with col2:
            s3_bucket = st.text_input("S3 Bucket Name:")
            s3_prefix = st.text_input("Prefix/Folder:", placeholder="mainframe/data/")
            s3_patterns = st.text_input("File Patterns:", placeholder="*.cbl,*.jcl,*.csv")
        
        if st.button("📥 Download from S3"):
            if all([aws_access_key, aws_secret_key, s3_bucket]):
                patterns = [p.strip() for p in s3_patterns.split(',') if p.strip()] if s3_patterns else ['*']
                
                data_source = DataSource(
                    source_type='s3',
                    connection_config={
                        'aws_access_key_id': aws_access_key,
                        'aws_secret_access_key': aws_secret_key,
                        'region': s3_region,
                        'bucket_name': s3_bucket,
                        'prefix': s3_prefix
                    },
                    file_patterns=patterns
                )
                
                with st.spinner("Connecting to S3 and downloading files..."):
                    upload_results = uploader.upload_from_s3(data_source)
            else:
                st.error("Please provide AWS credentials and bucket name")
    
    # HTTP URLs Upload
    elif upload_method == "HTTP URLs":
        st.subheader("🌍 HTTP URL Download")
        
        urls_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/mainframe/program.cbl\nhttps://example.com/data/employee.csv",
            height=150
        )
        
        if st.button("📥 Download from URLs"):
            if urls_text:
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                
                with st.spinner("Downloading files from URLs..."):
                    upload_results = uploader.upload_from_http_url(urls)
            else:
                st.error("Please enter at least one URL")
    
    # Database Export
    elif upload_method == "Database Export":
        st.subheader("🗄️ Database Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_type = st.selectbox("Database Type:", ["DB2", "Oracle", "SQL Server", "PostgreSQL"])
            db_host = st.text_input("Database Host:")
            db_port = st.number_input("Port:", value=50000 if db_type == "DB2" else 1521)
            db_name = st.text_input("Database Name:")
        
        with col2:
            db_username = st.text_input("Username:")
            db_password = st.text_input("Password:", type="password")
            db_schema = st.text_input("Schema (optional):")
            row_limit = st.number_input("Row Limit (0 = no limit):", value=10000, min_value=0)
        
        tables_text = st.text_area(
            "Tables to Export (one per line):",
            placeholder="EMPLOYEE_TABLE\nPAYROLL_TABLE\nCUSTOMER_TABLE",
            height=100
        )
        
        if st.button("📊 Export Database Tables"):
            if all([db_host, db_name, db_username, db_password, tables_text]):
                tables = [table.strip() for table in tables_text.split('\n') if table.strip()]
                
                config = {
                    'db_type': db_type,
                    'hostname': db_host,
                    'port': db_port,
                    'database': db_name,
                    'username': db_username,
                    'password': db_password,
                    'tables': tables
                }
                
                if db_schema:
                    config['schema'] = db_schema
                if row_limit > 0:
                    config['row_limit'] = row_limit
                
                data_source = DataSource(
                    source_type='database',
                    connection_config=config
                )
                
                with st.spinner("Connecting to database and exporting tables..."):
                    upload_results = uploader.upload_database_export(data_source)
            else:
                st.error("Please fill in all required database connection details and table names")
    
    # Display Upload Results
    if upload_results:
        st.subheader("📊 Upload Results")
        
        # Summary metrics
        successful = sum(1 for r in upload_results if r.success)
        failed = len(upload_results) - successful
        total_size = sum(r.file_size for r in upload_results if r.success)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", len(upload_results))
        with col2:
            st.metric("Successful", successful, delta=f"{(successful/len(upload_results)*100):.1f}%")
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        # Create results dataframe
        results_data = []
        for result in upload_results:
            results_data.append({
                'Status': '✅ Success' if result.success else '❌ Failed',
                'File Name': result.file_name,
                'File Type': result.detected_type,
                'Size (KB)': f"{result.file_size / 1024:.1f}" if result.success else 'N/A',
                'Error': result.error_message if not result.success else ''
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # File type distribution
        if successful > 0:
            type_counts = {}
            for result in upload_results:
                if result.success:
                    file_type = result.detected_type
                    type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            st.subheader("📈 File Type Distribution")
            
            import plotly.express as px
            
            type_df = pd.DataFrame(
                list(type_counts.items()),
                columns=['File Type', 'Count']
            )
            
            fig = px.pie(type_df, values='Count', names='File Type', 
                        title='Uploaded File Types')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show extracted files for archives
        extracted_info = []
        for result in upload_results:
            if result.success and 'extracted_files' in result.metadata:
                extracted_info.extend(result.metadata['extracted_files'])
        
        if extracted_info:
            st.subheader("📦 Extracted Archive Contents")
            st.write(f"Extracted {len(extracted_info)} files from archives:")
            for i, file_name in enumerate(extracted_info[:20], 1):  # Show first 20
                st.write(f"{i}. {file_name}")
            
            if len(extracted_info) > 20:
                st.write(f"... and {len(extracted_info) - 20} more files")
    
    # File Management Section
    st.subheader("📁 Uploaded Files Management")
    
    # Display upload statistics
    stats = uploader.get_upload_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Uploaded", stats['total_uploaded'])
    with col2:
        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
    with col3:
        st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")
    
    # File browser
    if st.checkbox("📂 Browse Uploaded Files"):
        file_type_filter = st.selectbox(
            "Filter by File Type:",
            ["All"] + list(set(uploader.file_type_mapping.values()))
        )
        
        filter_type = None if file_type_filter == "All" else file_type_filter
        uploaded_files = uploader.list_uploaded_files(filter_type)
        
        if uploaded_files:
            st.write(f"Found {len(uploaded_files)} files:")
            
            # Create file browser dataframe
            file_data = []
            for file_info in uploaded_files[:100]:  # Show first 100 files
                file_data.append({
                    'Name': file_info['name'],
                    'Type': file_info['type'],
                    'Size (KB)': f"{file_info['size'] / 1024:.1f}",
                    'Modified': file_info['modified'][:16],  # Truncate timestamp
                    'Extension': file_info['extension']
                })
            
            files_df = pd.DataFrame(file_data)
            st.dataframe(files_df, use_container_width=True)
            
            if len(uploaded_files) > 100:
                st.info(f"Showing first 100 files out of {len(uploaded_files)} total files")
        else:
            st.info("No uploaded files found")
    
    # Integration with Analysis System
    st.subheader("🔗 Integration with Analysis System")
    
    if st.button("🚀 Start Analysis with Uploaded Files"):
        # Get all uploaded source files
        source_files = uploader.list_uploaded_files()
        cobol_files = [f for f in source_files if f['type'] in ['COBOL_PROGRAM']]
        jcl_files = [f for f in source_files if f['type'] in ['JCL_JOB', 'JCL_PROC']]
        data_files = [f for f in source_files if f['type'] in ['DATA_CSV', 'DATA_EXCEL']]
        
        if cobol_files or jcl_files:
            st.success(f"Ready to analyze {len(cobol_files)} COBOL programs, {len(jcl_files)} JCL jobs, and {len(data_files)} data files")
            
            # Store file paths in session state for analysis
            st.session_state.analysis_files = {
                'source_files': [f['path'] for f in cobol_files + jcl_files],
                'data_files': [f['path'] for f in data_files]
            }
            
            st.info("Files prepared for analysis. Go to the Analysis tab to start intelligent analysis.")
        else:
            st.warning("No mainframe source files found. Please upload COBOL or JCL files first.")

# Data Upload Integration with Main Analysis System
class IntegratedUploadAnalysisSystem:
    """Integrated system that combines data upload with analysis"""
    
    def __init__(self):
        self.uploader = UniversalDataUploader()
        self.analysis_ready_files = {
            'source_files': [],
            'data_files': [],
            'config_files': []
        }
    
    def process_uploaded_files(self, upload_results: List[UploadResult]) -> Dict[str, List[str]]:
        """Process upload results and prepare files for analysis"""
        
        analysis_files = {
            'source_files': [],
            'data_files': [],
            'config_files': []
        }
        
        for result in upload_results:
            if not result.success:
                continue
            
            file_type = result.detected_type
            file_path = result.file_path
            
            # Categorize files for analysis
            if file_type in ['COBOL_PROGRAM', 'JCL_JOB', 'JCL_PROC', 'COPYBOOK', 'DB2_SQL', 'CICS_MAP']:
                analysis_files['source_files'].append(file_path)
            
            elif file_type in ['DATA_CSV', 'DATA_EXCEL', 'DATA_JSON']:
                analysis_files['data_files'].append(file_path)
            
            elif file_type in ['CONFIG']:
                analysis_files['config_files'].append(file_path)
        
        self.analysis_ready_files = analysis_files
        return analysis_files
    
    def auto_start_analysis(self, include_file_db2_comparison: bool = True) -> Dict[str, Any]:
        """Automatically start analysis with uploaded files"""
        
        if not self.analysis_ready_files['source_files']:
            return {'error': 'No source files available for analysis'}
        
        # Import the integrated analyzer
        from llm_intelligent_analysis import IntegratedMainframeAnalyzer
        
        # Initialize analyzer
        analyzer = IntegratedMainframeAnalyzer()
        
        # Prepare components for analysis
        components = []
        for file_path in self.analysis_ready_files['source_files']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            components.append({
                'name': Path(file_path).name,
                'path': file_path,
                'content': content,
                'component_type': 'PROGRAM',  # Will be auto-detected
                'analysis': {}
            })
        
        # Start comprehensive analysis
        csv_files = self.analysis_ready_files['data_files'] if include_file_db2_comparison else None
        
        results = analyzer.perform_comprehensive_analysis(components, csv_files)
        
        return results

# Example usage function
def example_complete_upload_flow():
    """Example showing complete upload and analysis flow"""
    
    # Initialize upload system
    uploader = UniversalDataUploader()
    
    # Example 1: Upload from local directory
    print("1. Uploading from local directory...")
    results = uploader.upload_from_local_directory(
        "/path/to/mainframe/source",
        recursive=True,
        file_patterns=['*.cbl', '*.jcl', '*.csv']
    )
    
    print(f"Uploaded {len(results)} files")
    
    # Example 2: Upload from FTP
    print("2. Uploading from FTP...")
    ftp_source = DataSource(
        source_type='ftp',
        connection_config={
            'host': 'ftp.mainframe.com',
            'username': 'user',
            'password': 'pass',
            'remote_directory': '/mainframe/source'
        },
        file_patterns=['*.cbl', '*.jcl']
    )
    
    ftp_results = uploader.upload_from_ftp(ftp_source)
    print(f"Downloaded {len(ftp_results)} files from FTP")
    
    # Example 3: Start integrated analysis
    print("3. Starting integrated analysis...")
    integrated_system = IntegratedUploadAnalysisSystem()
    
    all_results = results + ftp_results
    analysis_files = integrated_system.process_uploaded_files(all_results)
    
    analysis_results = integrated_system.auto_start_analysis(include_file_db2_comparison=True)
    
    print(f"Analysis completed for {len(analysis_files['source_files'])} source files")
    print(f"Found {len(analysis_results.get('component_analyses', {}))} component analyses")

if __name__ == "__main__":
    print("📤 Universal Data Upload & Ingestion System")
    print("=" * 60)
    
    print("Supported Upload Methods:")
    print("✅ Streamlit file upload interface")
    print("✅ Local directory scanning")
    print("✅ FTP/SFTP server download")
    print("✅ AWS S3 bucket download")
    print("✅ HTTP URL download")
    print("✅ Database table export")
    
    print("\\nSupported File Types:")
    print("📁 Mainframe Sources: COBOL, JCL, Copybooks, SQL, CICS")
    print("📊 Data Files: CSV, Excel, Sequential, VSAM")
    print("📄 Documentation: PDF, Word, Markdown")
    print("⚙️ Configuration: YAML, JSON, Properties")
    print("📦 Archives: ZIP, TAR, GZ")
    
    print("\\nKey Features:")
    print("🔍 Automatic file type detection")
    print("📂 Organized directory structure")
    print("📊 Upload statistics and monitoring")
    print("🔗 Direct integration with analysis system")
    print("📈 Progress tracking and error handling")
    print("🗂️ File browser and management")
    
    print("\\nTo use:")
    print("1. Choose upload method (Streamlit UI, FTP, S3, etc.)")
    print("2. Configure connection details if needed")
    print("3. Upload files - system auto-organizes by type")
    print("4. Review upload results and statistics")
    print("5. Start analysis directly with uploaded files")
    
    # Run example
    # example_complete_upload_flow()