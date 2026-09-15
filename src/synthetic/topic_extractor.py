"""Extract topics and keywords from the cleaned corpus for seeding QA generation.

Topics are extracted using simple statistical methods:
- High-frequency noun phrases (using regex patterns for Tamil)
- Named entities from metadata (URLs, publish dates, sections)
- Section headers from Wikipedia-like sources

These topics seed the Groq LLM to generate relevant, corpus-grounded
instruction-response pairs.

Run standalone:
    python -m src.synthetic.topic_extractor --input data/processed/text_clean.parquet \
        --output data/synthetic/topics.jsonl --max-topics 100
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set
from collections import Counter

from src.common.io_utils import read_parquet
from src.common.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_TOPICS = 100


def extract_question_seeds(text: str, max_length: int = 100) -> List[str]:
    """Extract potential question seeds from text.

    Simple heuristic: sentences ending with ? are questions.
    Also extract noun phrases and section headers.

    Args:
        text: Raw Tamil text.
        max_length: Maximum length of extracted phrase.

    Returns:
        List of question/topic strings.
    """
    seeds = []

    # Look for existing questions (lines ending with ?)
    for line in text.split("\n"):
        line = line.strip()
        if line.endswith("?") and len(line) < max_length:
            seeds.append(line.rstrip("?").strip())

    # Extract opening sentences (often topic sentences)
    sentences = text.split("।")  # Tamil period
    if len(sentences) > 0:
        first_sent = sentences[0].strip()
        if len(first_sent) > 20 and len(first_sent) < max_length:
            seeds.append(first_sent)

    return seeds


def extract_named_entities(record: Dict[str, Any]) -> List[str]:
    """Extract named entities and structured metadata as topics.

    Args:
        record: A corpus record with metadata fields.

    Returns:
        List of entity/topic strings.
    """
    entities = []

    # Extract title
    if "title" in record and record["title"] and isinstance(record["title"], str):
        title = record["title"].strip()
        if len(title) > 5:
            entities.append(title)

    # Extract section
    if "section" in record and record["section"] and isinstance(record["section"], str):
        section = record["section"].strip()
        if len(section) > 5 and section not in entities:
            entities.append(section)

    # Extract first few words from text as a topic
    if "text" in record and record["text"]:
        text = record["text"].strip()
        words = text.split()
        if len(words) >= 3:
            topic = " ".join(words[:5])  # First 5 words
            if topic not in entities:
                entities.append(topic)

    return entities


def extract_topics(
    corpus_records: List[Dict[str, Any]], max_topics: int = DEFAULT_MAX_TOPICS
) -> List[Dict[str, Any]]:
    """Extract diverse topics from corpus for QA seeding.

    Args:
        corpus_records: List of corpus text records.
        max_topics: Maximum number of unique topics to extract.

    Returns:
        List of topic records with topic string and source reference.
    """
    all_topics: Set[str] = set()
    topic_records: List[Dict[str, Any]] = []

    logger.info(f"Extracting topics from {len(corpus_records)} corpus records...")

    for i, record in enumerate(corpus_records):
        if i % 500 == 0:
            logger.info(f"  Processed {i}/{len(corpus_records)} records...")

        # Extract question seeds
        question_seeds = extract_question_seeds(record.get("text", ""))
        all_topics.update(question_seeds)

        # Extract named entities and metadata
        entities = extract_named_entities(record)
        all_topics.update(entities)

        if len(all_topics) >= max_topics:
            break

    logger.info(f"Extracted {len(all_topics)} unique topics from corpus")

    # Convert to records
    for topic in sorted(all_topics)[:max_topics]:
        if len(topic.strip()) > 5:  # Filter very short topics
            topic_records.append({
                "topic": topic,
                "source": "corpus",
                "language": "ta",
            })

    logger.info(f"Keeping {len(topic_records)} high-quality topics")
    return topic_records


def run(input_path: Path, output_path: Path, max_topics: int = DEFAULT_MAX_TOPICS) -> None:
    """Extract topics from corpus and save to JSONL.

    Args:
        input_path: Path to cleaned corpus Parquet file.
        output_path: Path to save topics as JSONL.
        max_topics: Maximum number of topics to extract.
    """
    logger.info(f"Loading corpus from {input_path}...")
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")

    topics = extract_topics(records, max_topics=max_topics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_record in topics:
            f.write(json.dumps(topic_record, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(topics)} topics to {output_path}")


def main() -> None:
    """CLI entrypoint: extract topics from corpus."""
    parser = argparse.ArgumentParser(description="Extract topics from cleaned corpus for QA generation seeding.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/text_clean.parquet"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic" / "topics.jsonl")
    parser.add_argument("--max-topics", type=int, default=DEFAULT_MAX_TOPICS)
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, max_topics=args.max_topics)


if __name__ == "__main__":
    main()
