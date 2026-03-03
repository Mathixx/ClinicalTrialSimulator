"""Run safety-mcp server via python -m safety_mcp."""

import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .server import mcp

if __name__ == "__main__":
    mcp.run(show_banner=False)

