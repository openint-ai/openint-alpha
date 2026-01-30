"""
Shared Configuration
Common configuration for all services
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "openint_data")

# Agent System Configuration
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")
AGENT_SYSTEM_PORT = int(os.getenv("AGENT_SYSTEM_PORT", "8001"))

# UI Backend Configuration
UI_BACKEND_PORT = int(os.getenv("UI_BACKEND_PORT", "3001"))
UI_FRONTEND_PORT = int(os.getenv("UI_FRONTEND_PORT", "3000"))

# Data Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "testdata"))

# Logging Configuration (debug logging off; use WARNING or INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
