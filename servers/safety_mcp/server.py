"""
safety-mcp: FastMCP server providing safety and regulatory tools.

Tools:
- search_pubmed: literature search via NCBI E-utilities.
- check_fda_labels: simplified label-derived limits for common drugs.
- verify_dose_limits: compare observed exposure against label limits.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import httpx
from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="[safety-mcp] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("safety_mcp")

mcp = FastMCP(name="safety-mcp")

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@mcp.tool(
    description="Search PubMed for articles. Returns titles, PMIDs, and optional abstracts for the query.",
)
def search_pubmed(query: str, max_results: int = 10, include_abstracts: bool = True) -> dict:
    results: list[dict] = []
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(
                f"{BASE}/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            )
            r.raise_for_status()
            data = r.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            logger.info("PubMed search: query=%r → 0 IDs", query)
            return {
                "query": query,
                "count": 0,
                "articles": [],
                "reasoning_trace": f"PubMed search: '{query}' returned no IDs.",
            }

        articles_by_id: dict[str, dict] = {
            pid: {"pmid": pid, "title": f"PMID {pid}", "snippet": ""} for pid in id_list
        }
        with httpx.Client(timeout=30.0) as client:
            r2 = client.get(
                f"{BASE}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"},
            )
            r2.raise_for_status()
            sum_data = r2.json()
        result = sum_data.get("result") or {}
        for pid in id_list:
            node = result.get(pid)
            if not isinstance(node, dict) or node.get("error") is not None:
                continue
            title = (node.get("title") or node.get("Title") or "").strip()
            if title:
                articles_by_id[pid]["title"] = title
            if include_abstracts:
                articles_by_id[pid]["snippet"] = (node.get("snippet") or node.get("abstract") or "")[:500]

        results = [articles_by_id[pid] for pid in id_list if pid in articles_by_id]

    except Exception as e:  # pragma: no cover
        logger.warning("PubMed search failed: query=%r error=%s", query, e)
        return {
            "query": query,
            "count": 0,
            "articles": [],
            "reasoning_trace": f"PubMed search failed: {e!s}.",
        }

    trace = f"Queried PubMed for '{query}'; found {len(results)} articles."
    return {"query": query, "count": len(results), "articles": results, "reasoning_trace": trace}


_FDA_LABELS: Dict[str, Dict[str, Any]] = {
    "warfarin": {
        "mtd_cmax_mg_L": 3.0,
        "therapeutic_range_mg_L": (1.0, 3.0),
        "black_box_warnings": ["Risk of major or fatal bleeding."],
    }
}


@mcp.tool(
    description=(
        "Look up simplified FDA label information for a drug: approximate therapeutic range, "
        "maximum tolerated Cmax, and black-box warnings."
    ),
)
def check_fda_labels(drug: str) -> dict:
    return _check_fda_labels_impl(drug)


def _check_fda_labels_impl(drug: str) -> dict:
    key = drug.strip().lower()
    data = _FDA_LABELS.get(key) or {
        "mtd_cmax_mg_L": 3.0,
        "therapeutic_range_mg_L": (1.0, 3.0),
        "black_box_warnings": [],
    }
    return {
        "drug": drug,
        "mtd_cmax_mg_L": float(data["mtd_cmax_mg_L"]),
        "therapeutic_range_mg_L": list(data["therapeutic_range_mg_L"]),
        "black_box_warnings": list(data["black_box_warnings"]),
    }


@mcp.tool(
    description="Compare observed Cmax/AUC against FDA-like limits and return a PASS/FAIL/WARNING safety verdict.",
)
def verify_dose_limits(
    drug: str,
    observed_cmax_mg_L: float,
    observed_auc_mg_h_L: float | None = None,
) -> dict:
    # NOTE: check_fda_labels is decorated into a Tool object; call internal impl here.
    label = _check_fda_labels_impl(drug)
    mtd = float(label["mtd_cmax_mg_L"])
    low, high = label["therapeutic_range_mg_L"]

    reasons: list[str] = []
    if observed_cmax_mg_L > mtd:
        verdict = "FAIL"
        reasons.append(f"Cmax {observed_cmax_mg_L:.2f} mg/L exceeds MTD {mtd:.2f} mg/L.")
    elif observed_cmax_mg_L > high:
        verdict = "WARNING"
        reasons.append(f"Cmax {observed_cmax_mg_L:.2f} mg/L above therapeutic upper bound {high:.2f} mg/L.")
    elif observed_cmax_mg_L < low:
        verdict = "WARNING"
        reasons.append(f"Cmax {observed_cmax_mg_L:.2f} mg/L below therapeutic lower bound {low:.2f} mg/L.")
    else:
        verdict = "PASS"
        reasons.append(f"Cmax {observed_cmax_mg_L:.2f} mg/L within therapeutic range {low:.2f}–{high:.2f} mg/L.")

    if observed_auc_mg_h_L is not None:
        reasons.append(f"Observed AUC: {observed_auc_mg_h_L:.2f} mg·h/L.")

    return {"drug": drug, "verdict": verdict, "reasoning": " ".join(reasons), "label": label}


if __name__ == "__main__":
    mcp.run(show_banner=False)

