"""PII redaction (phone numbers, emails) and Tamil profanity filtering.

Two distinct stages, logged separately to ``stats.json``:

1. **PII redaction** — regex-based; phone numbers and email addresses are
   replaced with placeholders (``[PHONE]`` / ``[EMAIL]``) in place. This does
   not drop records, since the rest of the document is usually still good
   training text once the PII is masked.
2. **Profanity filtering** — checks text against two open-source Tamil
   wordlists (see ``resources/``: native Tamil-script slurs from the Uli Slur
   List, and romanized "Tanglish" profanity from readme-SVG/Banned-words).

   Caveat worth flagging explicitly: several entries in the crowdsourced Uli
   list are ordinary dictionary words used *contextually* as slang/insults
   (e.g. "ஒன்பது" literally means "nine", "அத்தை" literally means "aunt").
   Blindly dropping every document that contains these tokens would remove a
   lot of perfectly good text. So by default this stage only **flags**
   matches (``has_profanity`` / ``profanity_matches`` columns) rather than
   dropping records; pass ``--drop-profanity`` to actually filter them out.

Run standalone:
    python -m src.filtering.pii_profanity --input data/raw/text_raw.parquet \
        --output data/interim/text_pii_profanity_filtered.parquet
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, RAW_DIR
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESOURCES_DIR = Path(__file__).parent / "resources"
NATIVE_PROFANITY_PATH = RESOURCES_DIR / "tamil_profanity_native.txt"
TANGLISH_PROFANITY_PATH = RESOURCES_DIR / "tamil_profanity_tanglish.txt"

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")
# Indian mobile numbers: 10 digits starting 6-9, optionally with a +91 / 0091 / 91 prefix.
PHONE_RE = re.compile(r"(?:\+91[\-\s]?|0091[\-\s]?|91[\-\s]?)?\b[6-9]\d{9}\b")


def load_wordlist(path: Path) -> List[str]:
    """Load a newline-delimited wordlist file, skipping blank lines and ``#`` comments.

    Args:
        path: Path to the wordlist ``.txt`` file.

    Returns:
        List of lowercased, stripped terms.
    """
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line.lower())
    return terms


def redact_pii(text: str) -> Tuple[str, int, int]:
    """Replace phone numbers and email addresses in ``text`` with placeholders.

    Args:
        text: Input text that may contain PII.

    Returns:
        A ``(redacted_text, phone_count, email_count)`` tuple.
    """
    email_count = len(EMAIL_RE.findall(text))
    redacted = EMAIL_RE.sub("[EMAIL]", text)
    phone_count = len(PHONE_RE.findall(redacted))
    redacted = PHONE_RE.sub("[PHONE]", redacted)
    return redacted, phone_count, email_count


def find_profanity_matches(text: str, wordlists: List[str]) -> List[str]:
    """Return the list of profanity terms found in ``text``.

    Single-word terms are matched against whitespace-delimited tokens (to
    avoid matching a banned word as a mere substring of an unrelated longer
    word); multi-word phrases are matched as substrings of the whole text.

    Args:
        text: Text to scan.
        wordlists: Flat list of lowercased profanity terms/phrases.

    Returns:
        List of matched terms (may contain duplicates if a term appears
        more than once).
    """
    lowered = text.lower()
    tokens = set(lowered.split())
    matches = []
    for term in wordlists:
        if " " in term:
            if term in lowered:
                matches.append(term)
        elif term in tokens:
            matches.append(term)
    return matches


def apply_pii_redaction(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """Redact PII from every record's text in place (no records are dropped).

    Args:
        records: List of text records (each must have a ``"text"`` field).

    Returns:
        A ``(records, total_phone_redactions, total_email_redactions)`` tuple.
    """
    total_phone, total_email = 0, 0
    out = []
    for record in records:
        text = record.get("text") or ""
        redacted, phone_count, email_count = redact_pii(text)
        record = dict(record)
        record["text"] = redacted
        record["pii_phone_redactions"] = phone_count
        record["pii_email_redactions"] = email_count
        record["pipeline_stage"] = "filtered.pii_redaction"
        total_phone += phone_count
        total_email += email_count
        out.append(record)
    return out, total_phone, total_email


def apply_profanity_filter(
    records: List[Dict[str, Any]],
    wordlists: List[str],
    drop: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Flag (and optionally drop) records containing profanity.

    Args:
        records: List of text records (each must have a ``"text"`` field).
        wordlists: Flat list of lowercased profanity terms/phrases to check against.
        drop: If ``True``, records with any match are moved to ``removed``.
            If ``False`` (default), every record is kept but annotated.

    Returns:
        A ``(kept, removed)`` tuple of record lists.
    """
    kept, removed = [], []
    for record in records:
        text = record.get("text") or ""
        matches = find_profanity_matches(text, wordlists)
        record = dict(record)
        record["has_profanity"] = len(matches) > 0
        record["profanity_matches"] = matches
        record["pipeline_stage"] = "filtered.profanity"
        if drop and matches:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def run(input_path: Path, output_path: Path, drop_profanity: bool = False) -> None:
    """Load records, redact PII, flag/drop profanity, and write the result.

    Logs two separate stats.json entries: ``filtering.pii_redaction`` and
    ``filtering.profanity``.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file.
        drop_profanity: Whether to actually drop flagged records (see module docstring).
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    redacted_records, total_phone, total_email = apply_pii_redaction(records)
    log_stage_stats(
        stage="filtering.pii_redaction",
        input_count=len(records),
        output_count=len(redacted_records),
        removed_count=0,
        extra={"phone_redactions": total_phone, "email_redactions": total_email},
    )
    logger.info("Redacted %d phone numbers and %d emails.", total_phone, total_email)

    wordlists = load_wordlist(NATIVE_PROFANITY_PATH) + load_wordlist(TANGLISH_PROFANITY_PATH)
    kept, removed = apply_profanity_filter(redacted_records, wordlists, drop=drop_profanity)
    flagged_count = sum(1 for r in kept + removed if r.get("has_profanity"))

    write_parquet(kept, output_path)
    logger.info(
        "Profanity: %d records flagged (drop_profanity=%s); wrote %d / %d records to %s",
        flagged_count,
        drop_profanity,
        len(kept),
        len(records),
        output_path,
    )

    log_stage_stats(
        stage="filtering.profanity",
        input_count=len(redacted_records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"drop_profanity": drop_profanity, "flagged_count": flagged_count, "wordlist_size": len(wordlists)},
    )


def main() -> None:
    """CLI entrypoint: apply PII redaction and profanity flagging/filtering."""
    parser = argparse.ArgumentParser(description="Redact PII and flag/filter Tamil profanity.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "text_raw.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_pii_profanity_filtered.parquet")
    parser.add_argument(
        "--drop-profanity",
        action="store_true",
        help="Drop records with any profanity match instead of just flagging them (see module docstring caveat).",
    )
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, drop_profanity=args.drop_profanity)


if __name__ == "__main__":
    main()
