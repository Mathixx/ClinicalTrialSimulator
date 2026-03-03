## Clinical Trial Simulator: Agentic AI for In‑Silico Clinical Discovery

An autonomous, multi-agent clinical trial simulator where an LLM “Lead Scientist” designs, executes, and interprets virtual trials using a network of MCP tools for pharmacokinetics (PK), safety, recruitment, and visualization. The system is built as a **Plan‑and‑Execute ReAct agent** that can iteratively refine trial designs based on quantitative evidence.

---

## Core Agentic Architecture

At the heart of this project is a **two‑stage agentic loop**:

- **Planner (Plan‑and‑Execute front‑end)**  
  - System prompt: `PLANNER_SYSTEM_PROMPT` (`agent.plan`)  
  - Input: natural‑language goal + tool manifest (MCP tools).  
  - Output: a `TrialProtocol` object with:
    - `goal`: normalized research question / trial goal.
    - `planned_steps`: ordered high‑level steps, each referencing one or more tools (e.g., “Run population PK simulation using simulate_pop_pk”, “Verify dose limits with safety_mcp.verify_dose_limits”, “Search PubMed for safety data on outliers”).

- **ReAct Executor (Reason‑and‑Act loop)**  
  - System prompt: `EXECUTOR_SYSTEM_PROMPT` (`agent.execute`).  
  - Loop structure:
    1. **Thought**: the agent explains what it will do next.
    2. **Action**: it calls exactly one MCP tool via function‑calling (e.g., PK simulation, safety check, recruitment, visualization).
    3. **Observation**: the tool’s JSON response (or error) is captured and written into a `TrialState`.
    4. **Pivot / Self‑Correction**: if observations indicate toxicity, implausible exposure, or feasibility issues, the agent can deviate from the original plan (e.g., insert extra safety checks, stress‑test edge phenotypes).
    5. **Completion**: once satisfied, the agent returns a free‑text scientific summary and closes the loop.

Internally, the orchestrator (`orchestrator.run_trial`) wraps this process and exposes it via CLI and FastAPI. All intermediate **ReAct cycles** (thought, action, observation) are stored in `TrialState.cycles` and surfaced in the frontend for developer inspection.

### Agent Flow (Mermaid)

```mermaid
flowchart TD
  U[Researcher goal<br/>(natural language)] --> P[Planner LLM<br/>Plan-and-Execute]
  P -->|TrialProtocol| S[TrialState]
  S --> R[ReAct Executor Loop]

  R --> T{Select MCP tool?}
  T -->|PK| PK[pk_sim_mcp<br/>population & PBPK models]
  T -->|Safety| SA[safety_mcp<br/>dose limits & literature]
  T -->|Recruitment| RE[recruitment_mcp<br/>feasibility & dropout]
  T -->|Visualization| VZ[viz_mcp<br/>PK & safety plots]

  PK --> O[Observation]
  SA --> O
  RE --> O
  VZ --> O

  O --> R
  R -->|Stop condition met| C[Summarize trial]
  C --> F[Structured TrialState JSON<br/>+ narrative insight]
```

The agent is **tool‑centric**: rather than embedding PK or safety logic into the LLM prompt, it calls dedicated scientific servers and uses their outputs as evidence to re‑plan.

---

## Scientific Capabilities

Within the current configuration (warfarin‑focused, but extensible), the system supports:

- **Virtual cohort construction & phenotype modeling**
  - Builds synthetic cohorts with user‑specified distributions over CYP2C9 genotypes and disease states.
  - Encodes demographic / phenotype traces alongside simulation outcomes for downstream analysis.

- **Mechanistic PK simulation**
  - One‑ and two‑compartment pharmacokinetic models with realistic parameters:
    - \(V_d\) (volume of distribution), \(k_e\) (elimination rate), \(CL\) (clearance), \(Q\) (inter‑compartmental flow).
  - Coarse PBPK model (gut → liver → central → kidney) to capture organ‑level effects:
    - Hepatic and renal function scaling.
    - Multiple dosing regimens with flexible dosing frequency.
  - Population PK:
    - Runs simulation per virtual patient.
    - Computes cohort‑level summaries of \(C_{\max}\), \(T_{\max}\), and AUC.

- **Non‑compartmental analysis (NCA)**
  - Computes:
    - \(C_{\max}\) (mg/L), \(T_{\max}\) (h).  
    - AUC via trapezoidal rule.  
    - Terminal half‑life via log‑linear regression on terminal points.
  - Flags data quality issues (e.g., too few time points).

- **Safety and toxicity assessment (via tools)**
  - Dose limit verification against configured safety corridors.
  - Literature‑driven safety checks (via `safety_mcp` server; e.g., PubMed / FDA label interrogations).
  - Outlier detection:
    - Patients with \(C_{\max}\) or AUC \(> 2 \times\) cohort median.
    - Aggregated reasons and per‑patient parameter traces.

- **Recruitment & feasibility analysis**
  - Evaluates enrollment and dropout risk given goal‑level constraints (via `recruitment_mcp`).
  - Enables “can this trial reasonably be run?” assessments, not just PK feasibility.

- **Visualization & narrative reporting**
  - PK concentration–time curves and safety corridors (thresholds, therapeutic windows).
  - Frontend trace viewer:
    - Human‑readable narrative summary (Markdown + Mermaid).
    - Per‑cycle ReAct trace with tool calls and observations.
    - Outlier tables and compact visual summaries (e.g., cohort vs. outlier concentration curves).

Overall, the system behaves like a **virtual clinical methods section + results section** that can be regenerated under new goals in a few seconds.

---

## Tech Stack

- **Language & runtime**
  - Python 3.11+
  - FastAPI backend, served alongside a lightweight HTML/JS frontend.

- **LLM & agent framework**
  - OpenAI‑compatible client (`openai` SDK), default model: **`gpt-4o-mini`** (configurable via `LLM_MODEL` and `OPENAI_BASE_URL`).
  - Custom, lightweight **planner + ReAct executor**:
    - `agent.plan`: JSON‑only planner for `TrialProtocol`.
    - `agent.execute`: ReAct loop over MCP tools with defensive nudging for proper tool usage and robust error handling.

- **MCP (Model Context Protocol) tooling**
  - `MCPManager` orchestrates MCP subprocesses based on `config/mcp_servers.json`.
  - Scientific MCP servers:
    - `pk_sim_mcp`: PK models, PBPK, population simulations, non‑compartmental analysis.
    - `safety_mcp`: safety rules, label / literature lookup, dose‑limit reasoning.
    - `recruitment_mcp`: recruitment and dropout feasibility.
    - `viz_mcp`: PK and safety visualizations (base64‑encoded PNGs).

- **Scientific Python libraries**
  - `numpy` for numerical arrays and statistics.
  - `scipy.integrate.solve_ivp` for ODE‑based PK / PBPK simulations.
  - `matplotlib` for server‑side visualizations (`viz_mcp`).

- **Frontend**
  - Vanilla HTML + JS with:
    - `Chart.js` for compact step‑level visualizations.
    - `marked` + `mermaid` for rendering Markdown and Mermaid diagrams from the agent’s narrative output.

---

## Installation

This project is managed with **uv**.

```bash
cd ClinicalTrialSimulator
uv sync --extra api --extra llm --group dev
```

Then configure your LLM:

```bash
cp .env.example .env
```

Set at minimum:

- `OPENAI_API_KEY` – API key for your OpenAI‑compatible endpoint.  
- Optionally: `OPENAI_BASE_URL`, `LLM_MODEL` (defaults to `gpt-4o-mini`).

> The MCP servers (PK, safety, recruitment, visualization) are launched automatically by the orchestrator when you run a trial; no manual server management is required for standard usage.

---

## Running a Discovery Loop

### 1. Launch full stack (API + UI)

From the repo root:

```bash
uv run python scripts/launch.py
```

Then open the UI in your browser:

```text
http://127.0.0.1:8000
```

In the UI:

- Enter a goal such as  
  *“Test 50mg warfarin on 80 elderly patients with varying CYP2C9. Flag anyone with \(C_{\max} > 3 \,\mathrm{mg/L}\) and stress‑test poor metabolizers.”*
- The system runs:
  1. **Planner** → constructs `TrialProtocol`.
  2. **ReAct executor** → iteratively calls PK, safety, recruitment, and visualization tools.
  3. **Summarizer** → renders a Markdown + Mermaid report and structured JSON state.
- Inspect:
  - High‑level narrative summary.
  - PK and safety plots.
  - Outlier table and parameter traces.
  - Full ReAct trace (per‑cycle thought, tool, arguments, and observation).

### 2. CLI: headless discovery from a goal

```bash
uv run run-trial "Test 50mg warfarin on 50 elderly with varying CYP2C9. Flag Cmax above 3 mg/L."
```

This invokes:

```python
from clinical_trial_simulator.orchestrator import run_trial

result = run_trial("Test 50mg warfarin on 50 elderly with varying CYP2C9. Flag Cmax above 3 mg/L.")
```

and returns a JSON‑serializable dict containing:

- `protocol`: planner output (`TrialProtocol`).
- `steps`: ReAct steps (tool calls + observations).
- `cycles`: higher‑level ReAct cycles.
- `results`: scientific aggregates (population PK summaries, outliers, etc.).

---

## Example Scientific Insight (Sample Output)

Below is a **representative** (simplified) agent summary from an in‑silico warfarin trial; numbers are illustrative.

> **Goal**  
> Simulate 80 virtual patients on warfarin 50 mg q24h with a mixed CYP2C9 genotype distribution (20% poor metabolizers, 30% intermediate, 50% normal). Flag any patient with \(C_{\max} > 3.5\,\mathrm{mg/L}\) and explore the effect of moderate hepatic impairment.
>
> **Agentic workflow**  
> 1. Planned a population PK study with phenotype‑stratified cohorts.  
> 2. Called `pk_sim_mcp.run_population_simulation` to generate time–concentration profiles and compute per‑patient \(C_{\max}\), \(T_{\max}\), and AUC.  
> 3. Performed non‑compartmental analysis on representative profiles to estimate half‑life and exposure.  
> 4. Queried `safety_mcp` for dose‑related bleeding risk at the observed exposure range.  
> 5. Visualized representative PK curves and outlier trajectories via `viz_mcp.plot_safety_corridors`.
>
> **Key findings**  
> - Median cohort \(C_{\max}\) was \(2.1\,\mathrm{mg/L}\); 11/80 patients (13.8%) exceeded the \(3.5\,\mathrm{mg/L}\) threshold.  
> - Outliers were strongly enriched in poor metabolizers with reduced hepatic function (\(\text{hepatic\_function} \le 0.6\)).  
> - Among outliers, the agent inferred an empirical exposure relationship consistent with  
>   \[
>     \mathrm{AUC} \approx \frac{\text{Dose}}{CL_{\text{effective}}}
>   \]
>   where \(CL_{\text{effective}}\) is reduced by both genotype and organ impairment.  
> - Safety tool calls indicated that this exposure range aligns with a significantly elevated bleeding risk in the literature.
>
> **Recommendation**  
> The agent recommends either:
> - Lowering the starting dose to 25 mg q24h for patients with combined poor metabolizer status and moderate hepatic impairment, **or**  
> - Maintaining 50 mg q24h but tightening monitoring criteria (e.g., more frequent INR checks and early dose titration).  
> In both cases, the projected reduction in outliers (patients above threshold) is \(> 50\%\) without materially compromising exposure in normal metabolizers.

Because the full `TrialState` is returned as JSON, this narrative is **reproducible**: all intermediate tool calls, PK trajectories, and cohort definitions can be re‑inspected or re‑analyzed downstream.

---
