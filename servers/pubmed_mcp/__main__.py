"""Run pubmed-mcp server via python -m pubmed_mcp."""

import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
# Reduce noise from MCP and httpx in the terminal
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .server import mcp

if __name__ == "__main__":
    mcp.run(show_banner=False)
