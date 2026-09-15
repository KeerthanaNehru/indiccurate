"""Streaming loader for Tamil Wikipedia and a Tamil OSCAR-derived web corpus.

Both sources are pulled via the Hugging Face ``datasets`` library in
streaming mode, so no full dataset download is required up front.

Sources:
    - Tamil Wikipedia: ``wikimedia/wikipedia`` (config ``20231101.ta``). This is
      the official, non-gated, parquet-backed Wikipedia dump distributed by
      Wikimedia on the Hugging Face Hub.
    - Tamil OSCAR subset: the *official* ``oscar-corpus/OSCAR-2301`` repo is
      gated behind a manual access request on the Hub, which can take days to
      be approved and would block this pipeline from running end-to-end. We
      instead default to ``AnanthZeke/oscar_tamil_2201``, a public,
      ungated re-upload of the Tamil portion of OSCAR-2201 on the Hub. Both
      the dataset name and config are CLI-configurable, so once you have
      approved access to the official gated repo you can point this script at
      it directly (set HF_TOKEN in .env for gated datasets).

Run standalone:
    python -m src.ingestion.hf_loader --limit 3000 --output data/raw/hf_text.parquet
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv

from src.common.io_utils import write_parquet
from src.common.paths import RAW_DIR
from src.common.schema import make_text_record
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WIKI_DATASET = "wikimedia/wikipedia"
DEFAULT_WIKI_CONFIG = "20231101.ta"
DEFAULT_WIKI_LICENSE = "CC-BY-SA-3.0"

DEFAULT_OSCAR_DATASET = "AnanthZeke/oscar_tamil_2201"
DEFAULT_OSCAR_CONFIG: Optional[str] = None
DEFAULT_OSCAR_LICENSE = "CC0-1.0 (community re-upload of OSCAR-2201 Tamil subset)"

MIN_CHARS = 20


def _stream_wikipedia(dataset: str, config: str, license_str: str, max_docs: Optional[int], hf_token: Optional[str]) -> Iterator[Dict[str, Any]]:
    """Yield schema-compliant text records streamed from the Tamil Wikipedia dataset.

    Args:
        dataset: HF dataset repo id, e.g. ``"wikimedia/wikipedia"``.
        config: HF dataset config/subset name, e.g. ``"20231101.ta"``.
        license_str: License string to attach to every record.
        max_docs: Stop after this many documents (``None`` = no limit).
        hf_token: Optional HF access token, for gated/rate-limited datasets.

    Yields:
        Text records in the shared schema (see ``src.common.schema``).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split="train", streaming=True, token=hf_token)
    if max_docs is not None:
        ds = ds.take(max_docs)  # streaming-friendly: only pulls the shards needed for N rows
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        yield make_text_record(
            text=text,
            source=f"hf:{dataset}:{config}",
            license=license_str,
            pipeline_stage="raw",
            doc_id=row.get("id"),
            title=row.get("title"),
            url=row.get("url"),
        )


def _stream_oscar(dataset: str, config: Optional[str], license_str: str, max_docs: Optional[int], hf_token: Optional[str]) -> Iterator[Dict[str, Any]]:
    """Yield schema-compliant text records streamed from the Tamil OSCAR-derived dataset.

    Args:
        dataset: HF dataset repo id.
        config: Optional HF dataset config name (``None`` if the dataset has no configs).
        license_str: License string to attach to every record.
        max_docs: Stop after this many documents (``None`` = no limit).
        hf_token: Optional HF access token, for gated/rate-limited datasets.

    Yields:
        Text records in the shared schema (see ``src.common.schema``).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split="train", streaming=True, token=hf_token)
    if max_docs is not None:
        ds = ds.take(max_docs)  # streaming-friendly: only pulls the shards needed for N rows
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        yield make_text_record(
            text=text,
            source=f"hf:{dataset}" + (f":{config}" if config else ""),
            license=license_str,
            pipeline_stage="raw",
        )


def run(
    output_path: Path,
    wiki_max_docs: Optional[int],
    oscar_max_docs: Optional[int],
    wiki_dataset: str = DEFAULT_WIKI_DATASET,
    wiki_config: str = DEFAULT_WIKI_CONFIG,
    oscar_dataset: str = DEFAULT_OSCAR_DATASET,
    oscar_config: Optional[str] = DEFAULT_OSCAR_CONFIG,
    hf_token: Optional[str] = None,
) -> None:
    """Stream Tamil Wikipedia + OSCAR-derived text and write them to a Parquet file.

    Args:
        output_path: Destination ``.parquet`` file.
        wiki_max_docs: Cap on documents pulled from Wikipedia (``None`` = unlimited).
        oscar_max_docs: Cap on documents pulled from the OSCAR-derived corpus (``None`` = unlimited).
        wiki_dataset: HF repo id for the Wikipedia source.
        wiki_config: HF config name for the Wikipedia source.
        oscar_dataset: HF repo id for the OSCAR-derived source.
        oscar_config: HF config name for the OSCAR-derived source, if any.
        hf_token: Optional HF access token read from the environment.
    """
    records = []

    logger.info("Streaming Tamil Wikipedia (%s / %s), cap=%s ...", wiki_dataset, wiki_config, wiki_max_docs)
    wiki_records = list(_stream_wikipedia(wiki_dataset, wiki_config, DEFAULT_WIKI_LICENSE, wiki_max_docs, hf_token))
    logger.info("Collected %d Wikipedia documents.", len(wiki_records))
    records.extend(wiki_records)

    logger.info("Streaming Tamil OSCAR-derived corpus (%s), cap=%s ...", oscar_dataset, oscar_max_docs)
    try:
        oscar_records = list(_stream_oscar(oscar_dataset, oscar_config, DEFAULT_OSCAR_LICENSE, oscar_max_docs, hf_token))
        logger.info("Collected %d OSCAR-derived documents.", len(oscar_records))
    except Exception as exc:  # noqa: BLE001 - keep pipeline resilient to a flaky/gated source
        logger.warning("Failed to stream OSCAR-derived corpus (%s): %s", oscar_dataset, exc)
        oscar_records = []
    records.extend(oscar_records)

    write_parquet(records, output_path)
    logger.info("Wrote %d total records to %s", len(records), output_path)

    log_stage_stats(
        stage="ingestion.hf_loader",
        input_count=len(wiki_records) + len(oscar_records),
        output_count=len(records),
        removed_count=0,
        extra={
            "wiki_dataset": f"{wiki_dataset}:{wiki_config}",
            "wiki_count": len(wiki_records),
            "wiki_max_docs": wiki_max_docs,
            "oscar_dataset": oscar_dataset,
            "oscar_count": len(oscar_records),
            "oscar_max_docs": oscar_max_docs,
        },
    )


def main() -> None:
    """CLI entrypoint: stream HF text sources and write them to a Parquet file."""
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description="Stream Tamil Wikipedia + OSCAR-derived text from Hugging Face.")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "hf_text.parquet", help="Output Parquet path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Total record cap combined across Wikipedia + OSCAR (demo/local-sized default; use -1 for unlimited).",
    )
    parser.add_argument(
        "--max-docs-per-source",
        type=int,
        default=None,
        help="Advanced override: exact cap per source, bypassing --limit's 50/50 split (use -1 for unlimited).",
    )
    parser.add_argument("--wiki-dataset", type=str, default=DEFAULT_WIKI_DATASET)
    parser.add_argument("--wiki-config", type=str, default=DEFAULT_WIKI_CONFIG)
    parser.add_argument("--oscar-dataset", type=str, default=DEFAULT_OSCAR_DATASET)
    parser.add_argument("--oscar-config", type=str, default=DEFAULT_OSCAR_CONFIG)
    args = parser.parse_args()

    if args.max_docs_per_source is not None:
        # Advanced path: same cap applied independently to each source.
        max_docs = None if args.max_docs_per_source < 0 else args.max_docs_per_source
        wiki_max_docs = oscar_max_docs = max_docs
    elif args.limit is not None and args.limit >= 0:
        # Default path: split the total combined --limit ~50/50 across the two sources.
        wiki_max_docs = args.limit // 2
        oscar_max_docs = args.limit - wiki_max_docs
    else:
        wiki_max_docs = oscar_max_docs = None

    hf_token = os.getenv("HF_TOKEN") or None

    run(
        output_path=args.output,
        wiki_max_docs=wiki_max_docs,
        oscar_max_docs=oscar_max_docs,
        wiki_dataset=args.wiki_dataset,
        wiki_config=args.wiki_config,
        oscar_dataset=args.oscar_dataset,
        oscar_config=args.oscar_config,
        hf_token=hf_token,
    )

    # `datasets`/`huggingface_hub` can leave non-daemon background threads
    # (e.g. Xet acceleration) alive after streaming finishes, which would
    # otherwise hang process exit indefinitely. Work is already flushed to
    # disk above, so an immediate hard exit here is safe.
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
