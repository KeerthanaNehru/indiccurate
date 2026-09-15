"""Shared funnel-stats logging used by every filtering/dedup/contamination stage.

Every stage that drops records should call :func:`log_stage_stats` exactly
once so that ``data/stats.json`` accumulates a full funnel history that the
Phase 7 Streamlit dashboard can turn into a funnel chart.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.paths import STATS_PATH, ensure_dirs


def log_stage_stats(
    stage: str,
    input_count: int,
    output_count: int,
    removed_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    stats_path: Path = STATS_PATH,
) -> Dict[str, Any]:
    """Append a run entry for ``stage`` to the shared stats.json file.

    Args:
        stage: Dotted stage identifier, e.g. ``"filtering.lang_id"``.
        input_count: Number of records the stage received.
        output_count: Number of records the stage produced (kept).
        removed_count: Number of records dropped. Computed as
            ``input_count - output_count`` if not supplied.
        extra: Any additional metadata to store with this run (thresholds used,
            per-reason breakdown, etc.).
        stats_path: Override for the stats file location (mainly for tests).

    Returns:
        The run entry dict that was appended.
    """
    ensure_dirs()
    if removed_count is None:
        removed_count = input_count - output_count

    entry: Dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_count": input_count,
        "output_count": output_count,
        "removed_count": removed_count,
        "extra": extra or {},
    }

    data = load_stats(stats_path)
    data.setdefault("runs", []).append(entry)
    stats_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def load_stats(stats_path: Path = STATS_PATH) -> Dict[str, Any]:
    """Load the stats.json file, returning ``{"runs": []}`` if it does not exist yet."""
    if not stats_path.exists():
        return {"runs": []}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)
