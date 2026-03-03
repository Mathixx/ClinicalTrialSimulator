from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _server_env() -> Dict[str, str]:
    env = dict(**__import__("os").environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT / "servers") + ":" + str(REPO_ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    return env


def _structured(result: Any) -> Any:
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "structuredContent") and getattr(result, "structuredContent") is not None:
        return getattr(result, "structuredContent")
    return None


@pytest.mark.anyio
async def test_pk_sim_mcp_lists_tools_and_runs_simulate_nca():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "pk_sim_mcp"],
        cwd=str(REPO_ROOT),
        env=_server_env(),
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout=20)
            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert "simulate_pk" in tool_names
            assert "simulate_nca" in tool_names
            assert "simulate_pbpk" in tool_names
            assert "simulate_pop_pk" in tool_names

            res = await session.call_tool(
                "simulate_nca",
                arguments={"t_hours": [0.0, 1.0, 2.0], "concentrations": [0.0, 2.0, 1.0]},
            )
            out = _structured(res)
            assert isinstance(out, dict)
            assert out["Cmax_mg_L"] == pytest.approx(2.0)


@pytest.mark.anyio
async def test_safety_mcp_lists_tools_and_runs_verify_dose_limits():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "safety_mcp"],
        cwd=str(REPO_ROOT),
        env=_server_env(),
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout=20)
            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert "check_fda_labels" in tool_names
            assert "verify_dose_limits" in tool_names
            assert "search_pubmed" in tool_names

            res = await session.call_tool(
                "verify_dose_limits",
                arguments={"drug": "warfarin", "observed_cmax_mg_L": 10.0},
            )
            out = _structured(res)
            assert out["verdict"] in ("FAIL", "WARNING", "PASS")
            assert out["verdict"] == "FAIL"


@pytest.mark.anyio
async def test_recruitment_mcp_lists_tools_and_runs_estimates():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "recruitment_mcp"],
        cwd=str(REPO_ROOT),
        env=_server_env(),
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout=20)
            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert "estimate_enrollment" in tool_names
            assert "calculate_dropout_risk" in tool_names

            res = await session.call_tool(
                "estimate_enrollment",
                arguments={"target_n": 100, "phenotype_prevalence": 0.1, "sites": 2},
            )
            out = _structured(res)
            assert out["estimated_months"] > 0


@pytest.mark.anyio
async def test_viz_mcp_lists_tools_and_runs_plot_pk_curve():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "viz_mcp"],
        cwd=str(REPO_ROOT),
        env=_server_env(),
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout=20)
            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert "plot_pk_curve" in tool_names
            assert "plot_safety_corridors" in tool_names

            res = await session.call_tool(
                "plot_pk_curve",
                arguments={"t_hours": [0, 1, 2], "concentrations": [0, 1, 0.5]},
            )
            out = _structured(res)
            assert isinstance(out, dict)
            assert isinstance(out.get("image_base64_png"), str)
            assert len(out["image_base64_png"]) > 100

