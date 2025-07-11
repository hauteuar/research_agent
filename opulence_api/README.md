# Opulence API

This repository contains the API codebase for the Opulence Deep Research Mainframe Agent system.

## Overview

The Opulence API provides endpoints and services for interacting with various analysis agents designed to understand and process mainframe artifacts. This includes capabilities for code parsing, chat-based interaction, data loading and schema mapping, lineage analysis, logic extraction, documentation generation, DB2 comparison, and vector indexing for semantic search.

This codebase is designed to be deployed as a standalone API service or integrated into a larger system. It communicates with underlying model servers via HTTP API calls, abstracting direct GPU management.

## Structure

-   `/`: Contains main API entry points (`api_main.py`), the central coordinator (`api_opulence_coordinator.py`), and the Streamlit UI application (`api_streamlit_app.py`).
-   `/agents`: Contains the API-specific versions of the different Opulence agents.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you intend to use the DB2 comparator, you may need to uncomment and install `ibm-db` and `ibm-db-sa` from `requirements.txt` and ensure DB2 drivers are correctly set up in your environment.*
    *The `vector_index_agent` uses a local CodeBERT model for embeddings. Ensure the model is available at the path specified in `vector_index_agent_api.py` (defaults to `./models/microsoft-codebert-base`) or update the path accordingly.*

3.  **Configuration:**
    The API system, particularly the `APIOpulenceCoordinator`, relies on configurations for model server endpoints. These are typically managed within `api_opulence_coordinator.py` or can be externalized. Refer to `APIOpulenceConfig` and `ModelServerConfig` classes for details. Environment variables might be used by specific agents (e.g., `db2_comparator_agent_api.py` for DB2 connection details).

## Running the API

The primary entry point for running the Opulence system with this API codebase is `api_main.py`.

```bash
python api_main.py --mode [web|batch|analyze|chat|server-status|test] [options]
```

Refer to `python api_main.py --help` for detailed options.

**Example Modes:**

*   **Web UI (Streamlit):**
    ```bash
    python api_main.py --mode web
    ```
    This will typically start the Streamlit application, which then uses the API coordinator.

*   **Batch Processing:**
    ```bash
    python api_main.py --mode batch --files path/to/your/file.cbl another/file.jcl
    ```

*   **Chat Mode (CLI):**
    ```bash
    python api_main.py --mode chat
    ```

**Model Server Endpoints:**
The `api_opulence_coordinator.py` and `api_main.py` expect model server endpoints to be configured. By default, they might look for servers like `http://localhost:8000`. Ensure your model serving infrastructure is running and accessible.

## Development

(Placeholder for development-specific instructions, e.g., running linters, tests, etc.)

## Contributing

(Placeholder for contribution guidelines.)
