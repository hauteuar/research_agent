# scripts/install.sh
#!/bin/bash
"""
Installation script for Opulence Deep Research Mainframe Agent
"""

set -e

echo "=========================================="
echo "Opulence Installation Script"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠ Warning: No NVIDIA GPU detected. CPU-only mode will be used."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv opulence_env
source opulence_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Opulence package
echo "Installing Opulence package..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs models cache config

# Copy default configuration
echo "Setting up configuration..."
if [ ! -f config/opulence_config.yaml ]; then
    cat > config/opulence_config.yaml << EOF
system:
  model_name: "codellama/CodeLlama-7b-Instruct-hf"
  max_tokens: 4096
  temperature: 0.1
  gpu_count: 3
  max_processing_time: 900
  batch_size: 32
  vector_dim: 768
  max_db_rows: 10000
  cache_ttl: 3600

db2:
  database: "TESTDB"
  hostname: "localhost"
  port: "50000"
  username: "db2user"
  password: "password"
  connection_timeout: 30

logging:
  level: "INFO"
  file: "logs/opulence.log"
  max_size_mb: 100
  backup_count: 5

security:
  enable_auth: false
  session_timeout: 3600
  allowed_file_types: [".cbl", ".cob", ".jcl", ".csv", ".ddl", ".sql", ".dcl", ".copy", ".cpy", ".zip"]

performance:
  enable_caching: true
  cache_size: 1000
  enable_gpu_monitoring: true
  health_check_interval: 60
  cleanup_interval: 300
EOF
fi

# Download model (optional)
echo "Would you like to download the default model now? (y/N)"
read -r download_model
if [[ $download_model =~ ^[Yy]$ ]]; then
    echo "Downloading CodeLlama model..."
    python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf'); AutoModel.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')"
fi

# Create startup scripts
cat > start_opulence.sh << EOF
#!/bin/bash
source opulence_env/bin/activate
python3 main.py --mode web
EOF

cat > start_batch.sh << EOF
#!/bin/bash
source opulence_env/bin/activate
python3 main.py --mode batch --folder "\$1"
EOF

chmod +x start_opulence.sh start_batch.sh

echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "To start Opulence:"
echo "  ./start_opulence.sh"
echo ""
echo "To run batch processing:"
echo "  ./start_batch.sh /path/to/files"
echo ""
echo "To analyze a component:"
echo "  source opulence_env/bin/activate"
echo "  python3 main.py --mode analyze --component FIELD_NAME --type field"
echo ""
echo "Web interface will be available at: http://localhost:8501"
echo ""
echo "Configuration file: config/opulence_config.yaml"
echo "Log files: logs/"
echo "Data directory: data/"
echo ""
echo "For more information, visit: