"""Review generator for ingested data."""
import json
from pathlib import Path


def build_review(output_dir: Path) -> dict:
    """Build review report for businesses in output directory."""
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            businesses = data.get("businesses", [])
        except Exception:
            businesses = []
    else:
        businesses = []

    total_rows = sum(b.get("rows", 0) for b in businesses)
    total_tables = sum(b.get("tables", 0) for b in businesses)
    review_data = {
        "businesses": businesses,
        "totals": {
            "businesses": len(businesses),
            "tables": total_tables,
            "rows": total_rows,
        },
    }
    return review_data
