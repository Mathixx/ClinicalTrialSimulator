from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import ToolDef

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "mcp_servers.json"
_MCP_INIT_TIMEOUT_SECONDS = 20.0
# PK/safety/recruitment can be slow; avoid closing the pipe before the server responds.
_MCP_CALL_TOOL_TIMEOUT_SECONDS = 300.0
# Visualization tools get a shorter limit so the run does not stall on slow plots.
_MCP_VIZ_CALL_TIMEOUT_SECONDS = 45.0

_VIZ_FAILED_MOCK_OBSERVATION = {
    "visualization_failed": True,
    "message": (
        "Visualization did not complete in time (or failed). "
        "Please ignore this failure; it is okay if there is nothing to show to the user. "
        "Continue with your summary."
    ),
}


def _load_mcp_config() -> Dict[str, Dict[str, Any]]:
    """Load MCP server commands from config, with sensible defaults."""
    if not _CONFIG_PATH.exists():
        return {
            "pk_sim_mcp": {
                "command": "python",
                "args": ["-m", "pk_sim_mcp"],
                "cwd": str(_REPO_ROOT),
            },
            "safety_mcp": {
                "command": "python",
                "args": ["-m", "safety_mcp"],
                "cwd": str(_REPO_ROOT),
            },
            "recruitment_mcp": {
                "command": "python",
                "args": ["-m", "recruitment_mcp"],
                "cwd": str(_REPO_ROOT),
            },
            "viz_mcp": {
                "command": "python",
                "args": ["-m", "viz_mcp"],
                "cwd": str(_REPO_ROOT),
            },
        }
    raw = json.loads(_CONFIG_PATH.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in raw.items():
        cwd = cfg.get("cwd", "${workspace_root}")
        if isinstance(cwd, str) and "${workspace_root}" in cwd:
            cwd = cwd.replace("${workspace_root}", str(_REPO_ROOT))
        out[name] = {
            "command": cfg.get("command", "python"),
            "args": cfg.get("args", []),
            "cwd": cwd or str(_REPO_ROOT),
        }
    return out


def _extract_tool_result(result: Any) -> Any:
    """Best-effort extraction of JSON from an MCP CallToolResult."""
    # Different SDK versions may expose structured content as structured_content or structuredContent.
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "structuredContent") and getattr(result, "structuredContent") is not None:
        return getattr(result, "structuredContent")
    content = getattr(result, "content", None)
    if content:
        for block in content:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                text = block.text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
    return {}


class MCPManager:
    """Manage MCP stdio sessions and expose tools to the agent."""

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._config = config or _load_mcp_config()
        self._sessions: Dict[str, Any] = {}
        self._tools: List[ToolDef] = []
        self._stack: Optional[AsyncExitStack] = None

    async def __aenter__(self) -> "MCPManager":
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as e:  # pragma: no cover - environment issue
            raise RuntimeError(
                "MCP SDK not installed. Install the 'mcp' package to run trials."
            ) from e

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(_REPO_ROOT / "servers"),
                str(_REPO_ROOT / "src"),
                env.get("PYTHONPATH", ""),
            ]
        )
        env["PYTHONUNBUFFERED"] = "1"

        # Start each server and initialize its session.
        for name, cfg in self._config.items():
            command = cfg.get("command", "python")
            args = list(cfg.get("args", []))
            if command == "python":
                command = sys.executable
                if "-u" not in args:
                    args = ["-u"] + args
            params = StdioServerParameters(
                command=command,
                args=args,
                cwd=cfg.get("cwd", str(_REPO_ROOT)),
                env=env,
            )
            read_stream, write_stream = await self._stack.enter_async_context(
                stdio_client(params)
            )
            session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await asyncio.wait_for(session.initialize(), timeout=_MCP_INIT_TIMEOUT_SECONDS)
            self._sessions[name] = session

        # Discover tools from each server.
        for server_name, session in self._sessions.items():
            resp = await session.list_tools()
            for tool in getattr(resp, "tools", []):
                schema = getattr(tool, "inputSchema", None) or {}
                self._tools.append(
                    ToolDef(
                        name=str(getattr(tool, "name", "")),
                        server=server_name,
                        description=str(getattr(tool, "description", "")),
                        parameters=schema,
                    )
                )

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._sessions.clear()
        self._tools.clear()
        self._stack = None

    @property
    def tools(self) -> List[ToolDef]:
        return list(self._tools)

    def tool_manifest(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool definitions derived from MCP tools."""
        manifest: List[Dict[str, Any]] = []
        for t in self._tools:
            params = t.parameters or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
            manifest.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"{t.server}_{t.name}",
                        "description": t.description,
                        "parameters": params,
                    },
                }
            )
        return manifest

    def _parse_tool_name(self, full_name: str) -> Tuple[str, str]:
        """Resolve full_name (server_tool or server.tool) to (server_name, tool_name)."""
        if "." in full_name:
            return full_name.split(".", 1)[0], full_name.split(".", 1)[1]
        # OpenAI-safe format: server_tool (e.g. pk_sim_mcp_simulate_pk).
        server_names = sorted(self._sessions.keys(), key=len, reverse=True)
        for server in server_names:
            prefix = server + "_"
            if full_name.startswith(prefix):
                return server, full_name[len(prefix) :]
        return self._resolve_short_name(full_name)

    def resolve_tool_server(self, full_name: str) -> str:
        """Return the MCP server name for a given tool full_name (for display)."""
        server, _ = self._parse_tool_name(full_name)
        return server

    def _is_viz_tool(self, server_name: str) -> bool:
        return server_name == "viz_mcp"

    async def call_tool(self, full_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by its fully-qualified name 'server_tool' or 'server.tool' and return JSON."""
        server_name, tool_name = self._parse_tool_name(full_name)

        session = self._sessions.get(server_name)
        if session is None:
            raise ValueError(f"Unknown MCP server: {server_name}")

        timeout = (
            _MCP_VIZ_CALL_TIMEOUT_SECONDS
            if self._is_viz_tool(server_name)
            else _MCP_CALL_TOOL_TIMEOUT_SECONDS
        )
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            if self._is_viz_tool(server_name):
                return dict(_VIZ_FAILED_MOCK_OBSERVATION)
            raise TimeoutError(
                f"Tool {full_name} did not respond within {timeout}s. "
                "The MCP server may have hung or crashed."
            ) from None
        except Exception as e:
            if self._is_viz_tool(server_name):
                return dict(_VIZ_FAILED_MOCK_OBSERVATION)
            raise

        try:
            return _extract_tool_result(result)
        except Exception as e:
            if self._is_viz_tool(server_name):
                return dict(_VIZ_FAILED_MOCK_OBSERVATION)
            raise

    def _resolve_short_name(self, short_name: str) -> Tuple[str, str]:
        matches = [t for t in self._tools if t.name == short_name]
        if not matches:
            raise ValueError(f"Unknown MCP tool: {short_name}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous MCP tool name {short_name!r}; use 'server.tool' form."
            )
        t = matches[0]
        return t.server, t.name

