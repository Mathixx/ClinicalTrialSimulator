# Clinical Trial Simulator

Production-grade, multi-agent clinical trial simulator using MCP for PK simulation and Safety Auditor (PubMed). Python 3.11+.

## Setup (uv)

From the repo root:

```bash
cd ClinicalTrialSimulator
uv sync --extra api --extra llm
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you use the goal → params / summary pipeline.

## Launch everything at once

Start the API and frontend (MCP servers are spawned by the orchestrator when you run a trial):

```bash
uv run python scripts/launch.py
```

Then open **http://127.0.0.1:8000**. The frontend shows a single prompt: describe your trial goal (e.g. *"Test 50mg Warfarin on 100 elderly with varying CYP2C9"*). The LLM turns it into params, the orchestrator runs the trial, and an LLM summary is shown. Use **Manual config** at the bottom to run a trial without the LLM.

**Why not `uv run launch`?** `uv run` can re-sync the env from the lockfile without installing the project itself, so the `launch` entry point may raise `ModuleNotFoundError: No module named 'clinical_trial_simulator'`. Running `uv run python scripts/launch.py` avoids that: the script adds the project to `sys.path` before importing.

## Commands (uv)

| Command | Description |
|--------|-------------|
| `uv run python scripts/launch.py` | Start API + frontend at http://127.0.0.1:8000 (recommended) |
| `uv run run-trial --cohort-size 10` | Run one trial with explicit args (no LLM) |
| `uv run run-trial-from-goal "Test 50mg Warfarin on 50 patients"` | Goal → LLM params → trial → LLM summary |

## Phase 1 (MCP servers, orchestrator)

- **pk-sim-mcp** (standalone): `PYTHONPATH=servers uv run python -m pk_sim_mcp`
- **pubmed-mcp** (standalone): `PYTHONPATH=servers uv run python -m pubmed_mcp`
- **Run trial**: `uv run run-trial --cohort-size 10` (orchestrator spawns MCP servers with correct paths)

If the orchestrator hangs, ensure the `mcp` (official SDK) and `fastmcp` packages are protocol-compatible; you can test each server with [MCP Inspector](https://github.com/modelcontextprotocol/inspector).

## LLM pipeline (goal → orchestrator → summary)

- **Setup**: `uv sync --extra api --extra llm` and set `OPENAI_API_KEY` in `.env` (optional: `OPENAI_BASE_URL` for proxy/Azure—use a full URL or host:port, `https://` is added if missing; `LLM_MODEL`; default model `gpt-4o-mini`).
- **CLI**: `uv run run-trial-from-goal "Test 50mg Warfarin on 50 elderly with varying CYP2C9"`
- **API**: `POST /run_trial_from_goal` with body `{"goal": "..."}` returns `params_used`, `result`, `summary`, `params_reasoning`.

## Full workflow

This section describes each step of a trial run, why it is done, and **where it is implemented** (file and main function).

1. **User goal (e.g. from the frontend)**  
   The user describes the trial in natural language (e.g. *"Test 50 mg Warfarin on 100 elderly with varying CYP2C9"*). This keeps the interface simple and allows the LLM to interpret intent.  
   **Where:** `frontend/index.html` (goal textarea, "Run trial from goal" button); on submit → `src/clinical_trial_simulator/api.py` → `post_run_trial_from_goal()` (POST `/run_trial_from_goal`), which starts a background thread.

2. **Goal → parameters (LLM, optional)**  
   An LLM turns the goal into structured parameters: dose, frequency, cohort size, drug, Cmax threshold. So we don’t hardcode trial designs; the same orchestrator supports many scenarios.  
   **Where:** `src/clinical_trial_simulator/run_trial_from_goal.py` → `run_trial_from_goal()` calls `goal_to_params(goal)` in `src/clinical_trial_simulator/llm_agents.py` (`goal_to_params`).

3. **Orchestrator loads MCP config**  
   The orchestrator reads `config/mcp_servers.json` (or uses defaults) to get the command, args, and cwd for pk-sim-mcp and pubmed-mcp. This allows different environments and installs without code changes.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_load_mcp_config()` (called at the start of `_run_trial_async()`). Config file: `config/mcp_servers.json`.

4. **Spawning pk-sim-mcp and pubmed-mcp**  
   Each MCP server is started as a subprocess with stdio transport.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_run_trial_async()` uses `stdio_client(pk_params)` and `stdio_client(pubmed_params)`. Server entry points: `servers/pk_sim_mcp/__main__.py`, `servers/pubmed_mcp/__main__.py` (run via `python -m pk_sim_mcp` / `python -m pubmed_mcp`).  
   The orchestrator sets `PYTHONPATH` and `PYTHONUNBUFFERED` so the servers can be found and their stdout is unbuffered. Unbuffered stdout is required so the MCP client receives responses immediately and doesn’t block forever at “Initializing MCP sessions”.

5. **Initializing MCP sessions**  
   The orchestrator runs the MCP handshake (`initialize`) with each server. If this step hangs, the server is likely not writing to stdout (e.g. due to buffering) or not starting correctly; check terminal logs (e.g. `[orchestrator]`, `[pk-sim-mcp]`).  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_run_trial_async()` inside `async with ClientSession(...) as pk_session` / `pubmed_session` → `await pk_session.initialize()` and `await pubmed_session.initialize()`.

6. **Building the cohort**  
   A synthetic cohort is built from phenotype distributions (e.g. CYP2C9). Each virtual patient gets PK parameters (e.g. 1- or 2-compartment) and a short reasoning trace. This gives a reproducible, interpretable population for the trial.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_build_cohort()` (called from `_run_trial_async()`). Per-patient params: `src/clinical_trial_simulator/phenotype_params.py` → `get_params_for_phenotypes()`.

7. **Running PK simulation per patient (pk-sim-mcp)**  
   For each patient, the orchestrator calls the `simulate_pk` tool on pk-sim-mcp with dose, frequency, and patient params. PK logic lives in the MCP server, not in the orchestrator, so we can swap or extend models without changing the core pipeline.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_run_trial_async()` loop → `pk_session.call_tool("simulate_pk", ...)`. Tool implementation: `servers/pk_sim_mcp/server.py` → `simulate_pk()` (and `servers/pk_sim_mcp/pk_models.py` for the simulation).

8. **Outlier detection**  
   Patients whose Cmax exceeds the threshold are flagged as outliers. These are passed to the Safety Auditor so we only query literature for at-risk cases.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_run_trial_async()`: in the same loop as step 7, when `c_max >= c_max_threshold_mg_L` the patient is appended to `outliers`.

9. **Safety Auditor (PubMed via pubmed-mcp)**  
   For each outlier, the orchestrator calls `search_pubmed` on pubmed-mcp with a safety-oriented query. The results (and optional reasoning trace) are attached to the outlier. This provides evidence-based context for high-exposure patients.  
   **Where:** `src/clinical_trial_simulator/run_trial.py` → `_run_trial_async()` loop over `outliers` → `pubmed_session.call_tool("search_pubmed", ...)`. Tool implementation: `servers/pubmed_mcp/server.py` → `search_pubmed()`.

10. **Optional LLM summary**  
    The structured result (per-patient summaries, outliers, PubMed snippets) can be summarized by an LLM for the user. The API returns both the raw result and the summary so the frontend can show either.  
   **Where:** `src/clinical_trial_simulator/run_trial_from_goal.py` → `run_trial_from_goal()` calls `summarize_trial_result(result)` in `src/clinical_trial_simulator/llm_agents.py` (`summarize_trial_result`).

11. **Response to the user**  
    The API (and frontend) expose status and result. The trial runs in a background thread with a timeout so the UI doesn’t block and the server doesn’t hang indefinitely if an MCP call stalls.  
   **Where:** `src/clinical_trial_simulator/api.py` → `_set_run_status()`, `get_run_trial_status()` (GET `/run_trial_status`). Frontend: `frontend/index.html` polls `/run_trial_status` and displays step, then result/summary when done. Trial timeout: `src/clinical_trial_simulator/run_trial.py` → `run_trial(timeout_seconds=...)` wraps `_run_trial_async` in `asyncio.wait_for()`.

**Debugging:** If pk-sim-mcp “doesn’t work” or “can’t be started”, watch the terminal for `[orchestrator]` and `[pk-sim-mcp]` logs. They show config, spawn command, session init, and the first `simulate_pk` call. If you see “Spawning pk-sim-mcp” but never “pk-sim-mcp session initialized OK”, the subprocess is likely failing to start (wrong cwd/PYTHONPATH) or its stdout is buffered; ensure `PYTHONUNBUFFERED=1` and `-u` in the server command.

## Config

MCP server commands: `config/mcp_servers.json`. The orchestrator spawns pk-sim-mcp and pubmed-mcp as subprocesses with the correct `PYTHONPATH`; replace `${workspace_root}` with the repo root when editing the config.
