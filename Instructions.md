# Clinical Trial Simulator

Autonomous, multi-agent clinical trial simulator where an LLM “Lead Scientist” plans and runs trials using MCP tools (PK, safety, recruitment, visualization). Python 3.11+.

## Setup (uv)

From the repo root:

```bash
cd ClinicalTrialSimulator
uv sync --extra api --extra llm --group dev
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you use the goal → params / summary pipeline.

## Launch everything at once (UI + API)

Start the API and frontend (MCP servers are spawned by the orchestrator when you run a trial):

```bash
uv run python scripts/launch.py
```

Then open **http://127.0.0.1:8000**.

- Type a goal like: *“Test 50mg Warfarin on 100 elderly with varying CYP2C9”*.
- The LLM planner creates a protocol, then the ReAct agent drives MCP tools (PK, safety, recruitment, viz).
- Use **Manual config** at the bottom if you want to bypass the LLM and just give simple parameters.

**Why not `uv run launch`?** `uv run` can re-sync the env from the lockfile without installing the project itself, so the `launch` entry point may raise `ModuleNotFoundError: No module named 'clinical_trial_simulator'`. Running `uv run python scripts/launch.py` avoids that: the script adds the project to `sys.path` before importing.

## Commands (uv, minimal)

| Command | Description |
|--------|-------------|
| `uv run python scripts/launch.py` | Start API + frontend at `http://127.0.0.1:8000` (recommended) |
| `uv run run-trial "Test 50mg Warfarin on 50 patients"` | Run one autonomous trial from a natural-language goal (CLI) |
| `uv run run-trial` | Same as above, with a built-in default goal |

## MCP servers

You normally **do not** need to start MCP servers manually; the orchestrator spawns them on demand.

For debugging, you can run them standalone:

- `PYTHONPATH=servers uv run python -m pk_sim_mcp` (PK models + NCA + population PK)
- `PYTHONPATH=servers uv run python -m safety_mcp` (PubMed + FDA label + dose checks)
- `PYTHONPATH=servers uv run python -m recruitment_mcp` (enrollment + dropout risk)
- `PYTHONPATH=servers uv run python -m viz_mcp` (PK / safety plots)

If things hang, ensure `mcp` and `fastmcp` are installed and protocol-compatible; you can also test servers with [MCP Inspector](https://github.com/modelcontextprotocol/inspector).

## LLM pipeline (goal → plan → ReAct → summary)

- **Setup**: `uv sync --extra api --extra llm` and set `OPENAI_API_KEY` in `.env` (optional: `OPENAI_BASE_URL`, `LLM_MODEL`; default model is `gpt-4o-mini`).
- **CLI**: `uv run run-trial "Test 50mg Warfarin on 50 elderly with varying CYP2C9"`  
  Uses `orchestrator.run_trial()` → planner (`agent.plan`) → ReAct executor (`agent.execute`) → returns a `TrialState` dict with protocol, steps, and summary.
- **API**: `POST /run_trial_from_goal` with body `{"goal": "..."}` (and optional `"dry_run": true`) returns a JSON object containing `state` (the full `TrialState`) and `summary`.

## Full workflow (short)

1. **User goal**  
   You send a natural-language goal from the frontend or CLI (no manual wiring of parameters needed).

2. **Planner (strategy)**  
   `agent.plan()` turns the goal + tool manifest into a `TrialProtocol` (a simple list of high-level steps).

3. **ReAct executor (autonomous loop)**  
   `agent.execute()` uses LLM tool-calling to iteratively choose which MCP tools to call (`pk-sim-mcp`, `safety-mcp`, `recruitment-mcp`, `viz-mcp`), based on observations. It can pivot if it sees toxicity or feasibility issues.

4. **State + summary**  
   The orchestrator returns a `TrialState` (protocol, steps, results, summary). The frontend shows the plan, intermediate thoughts/observations, and charts.

5. **Config**  
   MCP server commands live in `config/mcp_servers.json`. The orchestrator (`src/clinical_trial_simulator/orchestrator.py`) uses `MCPManager` (`mcp_manager.py`) to spawn and talk to all servers.

## Testing

Basic test commands:

- **Full test suite**:

  ```bash
  uv run pytest
  ```

- **Agent-specific tests**:

  ```bash
  uv run pytest tests/test_agent.py
  ```

## Config

MCP server commands: `config/mcp_servers.json`. The orchestrator spawns pk-sim-mcp and pubmed-mcp as subprocesses with the correct `PYTHONPATH`; replace `${workspace_root}` with the repo root when editing the config.
