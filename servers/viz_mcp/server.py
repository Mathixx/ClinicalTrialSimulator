from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastmcp import FastMCP

mcp = FastMCP(name="viz-mcp")


def _encode_figure() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


@mcp.tool(
    description=(
        "Plot PK concentration–time curve. "
        "Inputs: t_hours (x-axis) and concentrations (y-axis, mg/L) — must be same-length lists, e.g. "
        "representative_t_hours and representative_concentration_mg_L from simulate_pop_pk. "
        "Optional: label, threshold_mg_L. Returns base64 PNG or an error dict if input is invalid."
    ),
)
def plot_pk_curve(
    t_hours: List[float],
    concentrations: List[float],
    label: str | None = None,
    threshold_mg_L: float | None = None,
) -> Dict[str, Any]:
    if not t_hours or not concentrations:
        return {
            "error": "t_hours and concentrations must be non-empty lists.",
            "hint": "Use representative_t_hours and representative_concentration_mg_L from a prior simulate_pop_pk result.",
        }
    if len(t_hours) != len(concentrations):
        n = min(len(t_hours), len(concentrations))
        t_hours, concentrations = t_hours[:n], concentrations[:n]
        if n == 0:
            return {
                "error": "t_hours and concentrations had no overlap.",
                "hint": "Pass same-length lists from simulate_pop_pk or simulate_pk.",
            }

    plt.figure(figsize=(6, 3))
    plt.plot(t_hours, concentrations, label=label or "Concentration")
    if threshold_mg_L is not None:
        plt.axhline(threshold_mg_L, color="red", linestyle="--", label=f"Threshold {threshold_mg_L} mg/L")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (mg/L)")
    if label or threshold_mg_L is not None:
        plt.legend()
    plt.tight_layout()

    encoded = _encode_figure()
    return {
        "description": "PK concentration–time curve.",
        "image_base64_png": encoded,
    }


@mcp.tool(
    description=(
        "Plot PK curve with safety corridors: therapeutic range and MTD. "
        "Inputs: t_hours, concentrations (same-length lists from simulate_pop_pk or simulate_pk), "
        "therapeutic_min_mg_L, therapeutic_max_mg_L, mtd_mg_L. Returns base64 PNG or error dict."
    ),
)
def plot_safety_corridors(
    t_hours: List[float],
    concentrations: List[float],
    therapeutic_min_mg_L: float,
    therapeutic_max_mg_L: float,
    mtd_mg_L: float,
) -> Dict[str, Any]:
    if not t_hours or not concentrations:
        return {
            "error": "t_hours and concentrations must be non-empty lists.",
            "hint": "Use representative_t_hours and representative_concentration_mg_L from simulate_pop_pk.",
        }
    if len(t_hours) != len(concentrations):
        n = min(len(t_hours), len(concentrations))
        t_hours, concentrations = t_hours[:n], concentrations[:n]
        if n == 0:
            return {"error": "No overlap between t_hours and concentrations.", "hint": "Pass same-length lists."}

    plt.figure(figsize=(6, 3))
    plt.plot(t_hours, concentrations, label="Concentration")
    # Therapeutic window
    plt.fill_between(
        t_hours,
        therapeutic_min_mg_L,
        therapeutic_max_mg_L,
        color="green",
        alpha=0.15,
        label="Therapeutic range",
    )
    # MTD line
    plt.axhline(mtd_mg_L, color="red", linestyle="--", label="MTD")
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (mg/L)")
    plt.legend()
    plt.tight_layout()

    encoded = _encode_figure()
    return {
        "description": "PK curve with safety corridors (therapeutic range and MTD).",
        "image_base64_png": encoded,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)

