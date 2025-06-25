# README.md
"""
Opulence Deep Research Mainframe Agent
"""

# Opulence Deep Research Mainframe Agent 🧠

A comprehensive AI-powered system for analyzing mainframe COBOL, JCL, and legacy systems with deep research capabilities, field lineage tracking, and DB2 integration.

## 🌟 Features

### Core Capabilities
- **Multi-Agent Architecture**: 7 specialized agents working in parallel across 3 GPUs
- **Code Analysis**: Deep parsing of COBOL, JCL, CICS, and copybooks
- **Field Lineage Tracking**: Complete data flow analysis from creation to purge
- **Vector Search**: Semantic search across codebases using embeddings
- **DB2 Integration**: Compare data between DB2 tables and files (10K row limit)
- **Batch Processing**: Handle large volumes of files efficiently
- **Real-time Chat**: Interactive analysis through natural language

### Agent Breakdown
1. **Code Parser Agent**: Parses and chunks COBOL/JCL programs
2. **Vector Index Agent**: Creates searchable embeddings using FAISS/ChromaDB
3. **Data Loader Agent**: Handles CSV, DDL, DCLGEN file processing
4. **Lineage Analyzer Agent**: Tracks field usage and component lifecycle
5. **Logic Analyzer Agent**: Analyzes program logic with streaming support
6. **Documentation Agent**: Generates technical documentation
7. **DB2 Comparator Agent**: Compares SQLite and DB2 data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPUs (recommended: 3 GPUs)
- 16GB+ RAM
- CUDA 11.8+ (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/opulence/deep-research-agent.git
cd deep-research-agent

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Or manual installation
python3 -m venv opulence_env
source opulence_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quick Test

```bash
# Test the system
python3 scripts/test_system.py

# Start web interface
./start_opulence.sh
# Access at http://localhost:8501
```

## 💻 Usage

### Web Interface
```bash
python3 main.py --mode web
```

### Batch Processing
```bash
# Process folder of files
python3 main.py --mode batch --folder /path/to/cobol/files

# Process specific files
python3 main.py --mode batch --files program1.cbl program2.jcl data.csv
```

### Component Analysis
```bash
# Analyze a field
python3 main.py --mode analyze --component TRADE_DATE --type field

# Analyze a program
python3 main.py --mode analyze --component BKPG_TRD001 --type program

# Analyze with auto-detection
python3 main.py --mode analyze --component CUSTOMER_FILE
```

## 🔧 Configuration

Edit `config/opulence_config.yaml`:

```yaml
system:
  model_name: "codellama/CodeLlama-7b-Instruct-hf"
  gpu_count: 3
  max_processing_time: 900
  batch_size: 32

db2:
  hostname: "your-db2-host"
  port: "50000"
  username: "your-username"
  password: "your-password"
  database: "your-database"
```

## 📊 Example Queries

### Field Lineage Analysis
```
"Trace the lifecycle of TRADE_DATE field"
```

**Response**: Complete analysis showing:
- Where the field is defined (tables/copybooks)
- All programs that read/write/update the field
- Data transformations applied
- Lifecycle stages (creation, updates, archival, purge)
- Impact analysis for potential changes

### Component Analysis
```
"Analyze TRANSACTION_HISTORY_FILE lifecycle"
```

**Response**: Comprehensive report including:
- File creation points (which jobs/programs create it)
- All modules that use the file
- Data flow patterns
- Sample data and field meanings
- Record types and structures
- DB2 comparison if applicable

### Code Pattern Search
```
"Find programs that use security settlement logic"
```

**Response**: Semantic search results showing:
- Similar code patterns
- Relevant programs and paragraphs
- Business logic explanations
- Usage examples

## 🏗️ Architecture

### Multi-GPU Distribution
```
GPU 0: Code Parser + Logic Analyzer
GPU 1: Vector Index + Data Loader + DB2 Comparator  
GPU 2: Lineage Analyzer + Documentation Agent
```

### Data Flow
```
Files → Code Parser → Vector Index → Semantic Search
      ↓
   Data Loader → SQLite → DB2 Comparator
      ↓
  Lineage Analyzer → Impact Analysis → Documentation
```

## 📁 Project Structure

```
opulence/
├── agents/                 # Individual agent modules
│   ├── code_parser_agent.py
│   ├── vector_index_agent.py
│   ├── data_loader_agent.py
│   ├── lineage_analyzer_agent.py
│   ├── logic_analyzer_agent.py
│   ├── documentation_agent.py
│   └── db2_comparator_agent.py
├── utils/                  # Utility modules
│   ├── gpu_manager.py
│   ├── health_monitor.py
│   ├── cache_manager.py
│   └── config_manager.py
├── opulence_coordinator.py # Main coordinator
├── streamlit_app.py       # Web interface
├── main.py               # CLI entry point
├── config/               # Configuration files
├── data/                # Input data directory
├── logs/                # System logs
├── models/              # Downloaded models
└── cache/               # Cache directory
```

## 🔍 Key Features Deep Dive

### Field Lineage Tracking
- **Complete Lifecycle**: Track fields from creation to purge
- **Cross-Program Analysis**: See field usage across multiple programs
- **Impact Analysis**: Understand change implications
- **Data Quality**: Identify potential issues

### DB2 Integration
- **Data Comparison**: Compare SQLite and DB2 data (10K row limit)
- **Schema Analysis**: Analyze table structures
- **Data Validation**: Quality checks and reconciliation
- **Real-time Queries**: Direct DB2 connectivity

### Vector Search
- **Semantic Understanding**: Find similar code patterns
- **Business Logic Search**: Natural language queries
- **Code Similarity**: Identify duplicate or similar logic
- **Cross-Reference**: Link related components

## 🚀 Performance

### Processing Capabilities
- **Batch Processing**: 1000+ files in parallel
- **Response Time**: < 15 minutes for complex analysis
- **GPU Utilization**: Automatic load balancing
- **Memory Efficiency**: Intelligent caching

### Scalability
- **Horizontal Scaling**: Multi-GPU support
- **Vertical Scaling**: Configurable resource allocation
- **Cache Management**: LRU cache with TTL
- **Background Processing**: Non-blocking operations

## 🛠️ Development

### Running Tests
```bash
python3 scripts/test_system.py
```

### Docker Deployment
```bash
docker-compose up -d
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📚 Documentation

### API Reference
- Agent interfaces and methods
- Configuration options
- Extension points

### User Guide
- Detailed feature explanations
- Best practices
- Troubleshooting

### Examples
- Sample COBOL/JCL programs
- Analysis scenarios
- Integration patterns

## 🔒 Security

- **File Type Validation**: Only approved extensions
- **Input Sanitization**: Prevent injection attacks
- **Access Control**: Optional authentication
- **Data Encryption**: Sensitive information protection

## 📈 Monitoring

### Health Monitoring
- **System Resources**: CPU, memory, GPU usage
- **Performance Metrics**: Response times, throughput
- **Error Tracking**: Comprehensive logging
- **Alerting**: Threshold-based notifications

### Analytics Dashboard
- **Processing Statistics**: Files processed, success rates
- **Usage Patterns**: Popular queries, components
- **Performance Trends**: Historical analysis
- **Resource Utilization**: GPU and memory trends

## 🤝 Support

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: Community Q&A
- Wiki: Additional documentation

### Enterprise
- Professional support available
- Custom integrations
- Training and consulting
- SLA options

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with vLLM for efficient LLM inference
- FAISS and ChromaDB for vector search
- Streamlit for the web interface
- PyTorch ecosystem for ML capabilities

---

**Opulence Deep Research Mainframe Agent** - Transforming legacy system analysis with AI-powered insights.