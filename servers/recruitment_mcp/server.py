from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP

mcp = FastMCP(name="recruitment-mcp")


@mcp.tool(
    description=(
        "Estimate enrollment timeline for a trial based on target sample size, "
        "phenotype prevalence, and number of sites. Returns months to full enrollment "
        "and a coarse feasibility label."
    ),
)
def estimate_enrollment(
    target_n: int,
    phenotype_prevalence: float,
    sites: int = 1,
    monthly_screening_per_site: int = 20,
) -> Dict[str, Any]:
    """
    Estimate time to enroll target_n patients.

    phenotype_prevalence: fraction (0–1) of screened patients matching inclusion criteria.
    monthly_screening_per_site: how many patients can be screened per site per month.
    """
    if target_n <= 0 or sites <= 0 or monthly_screening_per_site <= 0:
        raise ValueError("target_n, sites, and monthly_screening_per_site must be positive.")
    eff_rate = max(phenotype_prevalence, 1e-4) * sites * monthly_screening_per_site
    months = float(target_n / eff_rate)
    if months <= 6:
        feasibility = "easy"
    elif months <= 18:
        feasibility = "moderate"
    else:
        feasibility = "hard"
    return {
        "target_n": target_n,
        "phenotype_prevalence": phenotype_prevalence,
        "sites": sites,
        "monthly_screening_per_site": monthly_screening_per_site,
        "estimated_months": months,
        "feasibility": feasibility,
    }


@mcp.tool(
    description=(
        "Estimate dropout risk based on trial duration, visit frequency, and invasiveness. "
        "Returns predicted dropout percentage and qualitative risk."
    ),
)
def calculate_dropout_risk(
    duration_weeks: float,
    visits_per_week: float,
    invasiveness: str = "moderate",
) -> Dict[str, Any]:
    """
    duration_weeks: planned treatment duration.
    visits_per_week: average scheduled visits per week (on-site or equivalent burden).
    invasiveness: 'low' | 'moderate' | 'high'.
    """
    if duration_weeks <= 0 or visits_per_week < 0:
        raise ValueError("duration_weeks must be positive and visits_per_week non-negative.")

    inv = invasiveness.strip().lower()
    base = {"low": 0.05, "moderate": 0.15, "high": 0.3}.get(inv, 0.15)
    # Simple heuristic: more visits and longer duration increase dropout risk.
    factor = 1.0 + 0.02 * max(duration_weeks - 4.0, 0.0) + 0.05 * max(visits_per_week - 1.0, 0.0)
    pct = min(base * factor, 0.9)

    if pct < 0.15:
        level = "low"
    elif pct < 0.35:
        level = "moderate"
    else:
        level = "high"

    return {
        "duration_weeks": duration_weeks,
        "visits_per_week": visits_per_week,
        "invasiveness": inv,
        "predicted_dropout_fraction": pct,
        "predicted_dropout_percent": pct * 100.0,
        "risk_level": level,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)

