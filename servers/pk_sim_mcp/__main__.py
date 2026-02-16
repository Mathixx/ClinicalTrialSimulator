"""Run pk-sim-mcp server via python -m pk_sim_mcp."""

import logging
import sys

# Before any MCP traffic: ensure no log output goes to stdout (MCP stdio uses stdout for JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
logging.getLogger("mcp").setLevel(logging.WARNING)

from .server import mcp

if __name__ == "__main__":
    # show_banner=False so no startup output can interfere with MCP stdio protocol
    mcp.run(show_banner=False)
