from pathlib import Path

# Cache directory for storing computed results (e.g., RSS)
CACHE_DIR = Path(__file__).parent.parent.parent / ".obs_cache"


def hello() -> str:
    return "Hello from obs!"
