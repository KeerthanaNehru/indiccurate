"""Shared record schema used across every pipeline stage.

Every record produced anywhere in IndicCurate — text or speech — is a plain
JSON-serializable dict that carries at minimum the fields declared in
``REQUIRED_TEXT_FIELDS`` / ``REQUIRED_SPEECH_FIELDS``. Downstream stages add
their own extra fields (e.g. ``lang_conf``, ``minhash_dup``) but must never
remove the base fields, and must update ``pipeline_stage`` to reflect the
stage that last touched the record.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable

REQUIRED_TEXT_FIELDS = ("text", "source", "license", "ingestion_timestamp", "pipeline_stage")
REQUIRED_SPEECH_FIELDS = ("audio_path", "source", "license", "ingestion_timestamp", "pipeline_stage")


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_text_record(
    text: str,
    source: str,
    license: str,
    pipeline_stage: str = "raw",
    **extra: Any,
) -> Dict[str, Any]:
    """Build a text record with all required base fields populated.

    Args:
        text: The raw or processed text content.
        source: Short identifier of the origin, e.g. ``"hf:wikipedia"``.
        license: License string, e.g. ``"CC-BY-SA-3.0"``.
        pipeline_stage: Tag naming the stage that produced this version of the record.
        **extra: Any additional fields (e.g. ``url``, ``title``, ``doc_id``).

    Returns:
        A dict containing the required fields plus any extras.
    """
    record: Dict[str, Any] = {
        "text": text,
        "source": source,
        "license": license,
        "ingestion_timestamp": now_iso(),
        "pipeline_stage": pipeline_stage,
    }
    record.update(extra)
    return record


def make_speech_record(
    audio_path: str,
    source: str,
    license: str,
    pipeline_stage: str = "raw",
    **extra: Any,
) -> Dict[str, Any]:
    """Build a speech record with all required base fields populated.

    Args:
        audio_path: Path to the audio file on disk (relative to project root).
        source: Short identifier of the origin, e.g. ``"openslr:slr65"``.
        license: License string, e.g. ``"CC-BY-4.0"``.
        pipeline_stage: Tag naming the stage that produced this version of the record.
        **extra: Any additional fields (e.g. ``transcript``, ``speaker_id``, ``duration_sec``).

    Returns:
        A dict containing the required fields plus any extras.
    """
    record: Dict[str, Any] = {
        "audio_path": audio_path,
        "source": source,
        "license": license,
        "ingestion_timestamp": now_iso(),
        "pipeline_stage": pipeline_stage,
    }
    record.update(extra)
    return record


def validate_records(records: Iterable[Dict[str, Any]], required_fields: Iterable[str]) -> None:
    """Raise ``ValueError`` if any record is missing a required field.

    Args:
        records: Iterable of record dicts to validate.
        required_fields: Field names that must be present (and non-None) in every record.

    Raises:
        ValueError: If any record is missing one of ``required_fields``.
    """
    required = tuple(required_fields)
    for i, record in enumerate(records):
        missing = [f for f in required if record.get(f) is None]
        if missing:
            raise ValueError(f"Record at index {i} is missing required fields: {missing}")
