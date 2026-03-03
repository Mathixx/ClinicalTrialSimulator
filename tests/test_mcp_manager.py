from __future__ import annotations

import asyncio

import pytest

from clinical_trial_simulator.mcp_manager import MCPManager


@pytest.mark.anyio
async def test_mcp_manager_discovers_tools_from_all_servers():
    async with MCPManager() as mcp:
        tools = mcp.tools
        assert tools, "Expected at least one MCP tool to be discovered."
        servers = {t.server for t in tools}
        assert "pk_sim_mcp" in servers
        assert "safety_mcp" in servers
        assert "recruitment_mcp" in servers
        assert "viz_mcp" in servers

        manifest = mcp.tool_manifest()
        assert any(d["function"]["name"] == "pk_sim_mcp_simulate_pk" for d in manifest)
        assert any(d["function"]["name"] == "safety_mcp_verify_dose_limits" for d in manifest)

