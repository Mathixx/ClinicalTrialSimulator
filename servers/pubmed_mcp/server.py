"""
pubmed-mcp: FastMCP server wrapping NCBI E-utilities for PubMed search.
Used by the Safety Auditor agent to cross-reference simulation results with literature.
"""

from __future__ import annotations

import logging
import sys

import httpx
from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="[pubmed-mcp] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("pubmed_mcp")

mcp = FastMCP(name="pubmed-mcp")

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@mcp.tool(
    description="Search PubMed for articles. Returns titles, PMIDs, and optional abstracts for the query.",
)
def search_pubmed(query: str, max_results: int = 10, include_abstracts: bool = True) -> dict:
    """
    query: PubMed search query (e.g. 'CYP2C9 warfarin toxicity genotype').
    max_results: maximum number of articles to return (default 10).
    include_abstracts: if True, fetch abstracts via efetch (default True).
    """
    results: list[dict] = []
    try:
        # esearch
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
            return {"query": query, "count": 0, "articles": [], "reasoning_trace": f"PubMed search: '{query}' returned no IDs."}

        # esummary for JSON metadata (title, etc.). Result keys are PMIDs.
        articles_by_id: dict[str, dict] = {pid: {"pmid": pid, "title": f"PMID {pid}", "snippet": ""} for pid in id_list}
        if id_list:
            with httpx.Client(timeout=30.0) as client:
                r2 = client.get(
                    f"{BASE}/esummary.fcgi",
                    params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"},
                )
                r2.raise_for_status()
                sum_data = r2.json()
            result = sum_data.get("result") or {}
            for pid in id_list:
                if pid not in articles_by_id:
                    continue
                node = result.get(pid)
                if isinstance(node, dict) and node.get("error") is None:
                    title = (node.get("title") or node.get("Title") or "").strip()
                    if title:
                        articles_by_id[pid]["title"] = title
                    if include_abstracts:
                        articles_by_id[pid]["snippet"] = (node.get("snippet") or node.get("abstract") or "")[:500]

        results = [articles_by_id[pid] for pid in id_list if pid in articles_by_id]

    except Exception as e:
        logger.warning("PubMed search failed: query=%r error=%s", query, e)
        return {
            "query": query,
            "count": 0,
            "articles": [],
            "reasoning_trace": f"PubMed search failed: {e!s}.",
        }

    logger.info("PubMed search: query=%r → %s articles (IDs: %s)", query, len(results), [a.get("pmid") for a in results])
    trace = f"Queried PubMed for '{query}'; found {len(results)} articles."
    return {
        "query": query,
        "count": len(results),
        "articles": results,
        "reasoning_trace": trace,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
