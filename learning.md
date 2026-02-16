### Debugging: Port already in use (e.g. 8000)

- **Context:** Needed when restarting the Clinical Trial Simulator or when a previous run didn’t exit cleanly; the frontend/API won’t start until the port is free.
- **How to check who is using the port (macOS/Linux):**
  - List process using port 8000:  
    `lsof -i :8000`  
  - Or with `netstat`:  
    `netstat -anv | grep 8000`
- **How to solve it:**
  - Kill the process by PID (from the second column of `lsof -i :8000`):  
    `kill <PID>`  
  - Force kill if it doesn’t exit:  
    `kill -9 <PID>`  
---

### Trial run stuck at MCP startup (progress + timeout)

- **Concept:** The orchestrator spawns pk-sim-mcp and pubmed-mcp over stdio; the official MCP Python SDK client can block indefinitely during session initialize or the first tool call if there is a protocol mismatch with FastMCP, so the run appears “stuck” after the FastMCP banners.
- **Context:** Users see “Spawning pk-sim-mcp and pubmed-mcp…” or “Initializing MCP sessions…” for minutes with no progress. The frontend now shows the current step and the backend enforces a timeout so the run does not hang forever.
- **What we did:**
  - **Progress:** The backend reports steps (e.g. “Interpreting goal with LLM…”, “Running PK simulation (patient 3/30)…”, “Safety Auditor: querying PubMed…”). POST `/run_trial_from_goal` starts a background run; the frontend polls GET `/run_trial_status` and displays the current `step` with a spinner until `status` is `done` or `error`.
  - **Timeout:** `run_trial()` wraps the async run in `asyncio.wait_for(..., timeout=120)`. If the MCP handshake or simulation exceeds 120 seconds, the run returns an error dict (`result.error`) and the summarizer still runs; the frontend shows the timeout message.
- **Tuning:** Timeout is set by `DEFAULT_TRIAL_TIMEOUT_SECONDS` in `run_trial.py`; pass `timeout_seconds=None` to disable (not recommended in production).

### Why the client blocks at “Starting MCP server 'pk-sim-mcp'” / “Initializing MCP sessions”

- **Concept:** The orchestrator (MCP client) spawns pk-sim-mcp and pubmed-mcp as subprocesses and talks to them over stdio (stdin/stdout). The client sends an `initialize` request and waits for a response. If the **server’s stdout is block-buffered**, or if the server writes anything other than JSON-RPC to stdout (e.g. a banner or log line), the client never gets a valid response and blocks until the trial timeout, then you see `BrokenResourceError` when the task group tears down.
- **Context:** You see the FastMCP “Starting MCP server” log (from the server) then nothing; after ~60s you get “unhandled errors in a TaskGroup” / `BrokenResourceError`. That usually means the **trial timeout** fired (e.g. 60s) while waiting for `initialize()`; the server may have been blocking at startup or writing to stdout.
- **Fixes applied:**
  - **Use `async with ClientSession(read_stream, write_stream) as session`** before `await session.initialize()` — the MCP SDK's `BaseSession.__aenter__` starts the receive loop that reads server responses; without it the client never sees the response and blocks forever.
  - Orchestrator: (1) `PYTHONUNBUFFERED=1` and `-u` when spawning servers; (2) spawn with `sys.executable` so the server uses the same venv; (3) `show_banner=False` in the servers so no startup banner runs; (4) servers’ `__main__.py` calls `logging.basicConfig(stream=sys.stderr, ...)` so no log goes to stdout; (5) dedicated `MCP_INIT_TIMEOUT_SECONDS` (20s) so we fail fast with a clear “did not respond to initialize” error instead of waiting the full trial timeout.
  - If you still block, the cause may be a protocol/version mismatch between the official MCP SDK client and FastMCP (try MCP Inspector to confirm the server responds correctly).
---

