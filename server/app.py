"""
Disaster Response Coordination Environment — FastAPI Server App
Wraps DisasterResponseEnvironment and registers OpenEnv HTTP/WebSocket routes.
"""
import os
import sys

# Make parent importable inside Docker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app
from models import DisasterAction, DisasterObservation
from server.disaster_environment import DisasterResponseEnvironment


def create_environment() -> DisasterResponseEnvironment:
    """Factory: each WebSocket client gets its own isolated environment instance."""
    task_level = int(os.environ.get("TASK_LEVEL", "1"))
    return DisasterResponseEnvironment(task_level=task_level)


# Build the FastAPI application
app = create_app(
    create_environment,   # factory — one instance per session
    DisasterAction,
    DisasterObservation,
    env_name="disaster_response_env",
)
