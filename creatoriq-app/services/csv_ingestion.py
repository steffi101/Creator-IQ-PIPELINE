"""
CSV Ingestion Service

Parses uploaded CSV files containing campaign performance data.
Validates schema, cleans data, and returns structured analysis.
"""

import csv
import io
from typing import Optional


EXPECTED_COLUMNS = {
    "required": ["hook_style", "cpi"],
    "optional": ["format", "ctr", "completion_rate", "installs", "spend", "days_live", "audience"],
}


def parse_csv(content: str) -> dict:
    """Parse CSV content and return structured data."""
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    errors = []

    headers = reader.fieldnames or []
    headers_lower = [h.lower().strip() for h in headers]

    # Validate required columns
    missing = [col for col in EXPECTED_COLUMNS["required"] if col not in headers_lower]
    if missing:
        return {"error": f"Missing required columns: {', '.join(missing)}", "rows": [], "headers": headers_lower}

    for i, row in enumerate(reader):
        cleaned = {}
        for key, value in row.items():
            key_lower = key.lower().strip()
            if value is None or value.strip() == "":
                cleaned[key_lower] = None
            else:
                try:
                    cleaned[key_lower] = float(value) if key_lower in ["cpi", "ctr", "completion_rate", "spend"] else value.strip()
                    if key_lower in ["installs", "days_live"]:
                        cleaned[key_lower] = int(float(value))
                except ValueError:
                    cleaned[key_lower] = value.strip()
        rows.append(cleaned)

    return {
        "headers": headers_lower,
        "rows": rows,
        "row_count": len(rows),
        "error": None,
    }


def generate_performance_summary_text(summary: dict) -> str:
    """Convert a performance summary dict into natural language for the LLM."""
    overall = summary.get("overall", {})
    by_hook = summary.get("by_hook_style", [])
    by_format = summary.get("by_format", [])
    by_audience = summary.get("by_audience", [])
    best = summary.get("best_performers", [])
    worst = summary.get("worst_performers", [])

    text = "=== REAL CAMPAIGN PERFORMANCE DATA ===\n\n"

    text += f"OVERALL METRICS:\n"
    text += f"  Total Creatives Tested: {overall.get('total_creatives', 0)}\n"
    text += f"  Average CPI: ${overall.get('avg_cpi', 0):.2f}\n"
    text += f"  Average CTR: {overall.get('avg_ctr', 0):.2f}%\n"
    text += f"  Average Completion Rate: {overall.get('avg_completion', 0):.1f}%\n"
    text += f"  Total Installs: {overall.get('total_installs', 0)}\n"
    text += f"  Total Spend: ${overall.get('total_spend', 0):.2f}\n"
    text += f"  Best CPI: ${overall.get('best_cpi', 0):.2f}\n"
    text += f"  Worst CPI: ${overall.get('worst_cpi', 0):.2f}\n\n"

    if by_hook:
        text += "PERFORMANCE BY HOOK STYLE (sorted best to worst):\n"
        for h in by_hook:
            text += f"  {h['hook_style']}: avg CPI ${h['avg_cpi']:.2f}, avg CTR {h['avg_ctr']:.2f}%, {h['count']} creatives, {h['installs']} installs\n"
        text += "\n"

    if by_format:
        text += "PERFORMANCE BY FORMAT:\n"
        for f in by_format:
            text += f"  {f['format']}: avg CPI ${f['avg_cpi']:.2f}, avg CTR {f['avg_ctr']:.2f}%, {f['count']} creatives\n"
        text += "\n"

    if by_audience:
        text += "PERFORMANCE BY AUDIENCE:\n"
        for a in by_audience:
            text += f"  {a['audience']}: avg CPI ${a['avg_cpi']:.2f}, {a['installs']} installs\n"
        text += "\n"

    if best:
        text += "TOP 3 BEST PERFORMERS:\n"
        for b in best:
            text += f"  {b.get('hook_style', '?')} + {b.get('format', '?')}: CPI ${b.get('cpi', 0):.2f}, CTR {b.get('ctr', 0):.1f}%, {b.get('installs', 0)} installs\n"
        text += "\n"

    if worst:
        text += "BOTTOM 3 WORST PERFORMERS:\n"
        for w in worst:
            text += f"  {w.get('hook_style', '?')} + {w.get('format', '?')}: CPI ${w.get('cpi', 0):.2f}, CTR {w.get('ctr', 0):.1f}%, {w.get('installs', 0)} installs\n"

    return text
