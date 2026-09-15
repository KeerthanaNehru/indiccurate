"""Parquet / JSONL read-write helpers shared by every pipeline stage.

Per the project convention: large intermediate outputs are stored as Parquet
(via pyarrow), small/debug outputs as JSONL.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of record dicts to a Parquet file, creating parent dirs.

    Args:
        records: List of JSON-serializable dicts, all sharing a compatible schema.
        path: Destination ``.parquet`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame.

    Args:
        path: Source ``.parquet`` file path.

    Returns:
        The loaded DataFrame.
    """
    return pq.read_table(path).to_pandas()


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write an iterable of record dicts to a JSONL file, creating parent dirs.

    Args:
        records: Iterable of JSON-serializable dicts.
        path: Destination ``.jsonl`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append a single record dict to a JSONL file, creating parent dirs.

    Args:
        record: JSON-serializable dict.
        path: Destination ``.jsonl`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Lazily read a JSONL file, yielding one dict per non-empty line.

    Args:
        path: Source ``.jsonl`` file path.

    Yields:
        Each parsed record dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
