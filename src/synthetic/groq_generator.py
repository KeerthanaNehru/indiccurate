"""Generate instruction-response pairs using Groq LLM API (free tier).

Uses the Groq API with Llama models to generate Tamil instruction-response pairs
from corpus-extracted topics. Implements rate limiting to stay within free tier
constraints (e.g., 30 requests/minute for some models).

Run standalone:
    python -m src.synthetic.groq_generator --topics data/synthetic/topics.jsonl \
        --output data/synthetic/qa_pairs_raw.jsonl --max-pairs 100
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_PAIRS = 100
DEFAULT_BATCH_SIZE = 5
DEFAULT_MODEL = "llama-3.1-8b-instant"  # Fast, free-tier model
DEFAULT_RATE_LIMIT_DELAY = 0.5  # Seconds between requests


def get_groq_client():
    """Initialize Groq client from GROQ_API_KEY environment variable.

    Returns:
        Groq client instance, or None if API key not set.
    """
    try:
        from groq import Groq
        import os

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set in environment. Set it in .env or export it.")
            return None

        # Initialize without proxy parameters (they may not be supported in all versions)
        return Groq(api_key=api_key)
    except ImportError:
        logger.error("groq library not installed. Install with: pip install groq")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


def generate_qa_pair(
    client,
    topic: str,
    model: str = DEFAULT_MODEL,
    language: str = "ta",
) -> Optional[Dict[str, Any]]:
    """Generate a single instruction-response pair from a topic using Groq.

    Args:
        client: Groq client instance.
        topic: Topic string to seed generation.
        model: Groq model ID to use.
        language: Language code (used in prompt).

    Returns:
        Dict with 'instruction' and 'response' keys, or None on error.
    """
    prompt = f"""Generate a concise instruction-response pair in Tamil (ta) based on this topic:

Topic: {topic}

Instructions:
1. Create a clear, answerable instruction (question or task) in Tamil
2. Provide a detailed, accurate response in Tamil
3. Keep response under 500 words
4. Use natural, conversational Tamil

Format your response as JSON with 'instruction' and 'response' keys.
Only output valid JSON, no other text.

Example format:
{{"instruction": "உதாரணம் கேள்வி?", "response": "விस்તარం उत्तर."}}

Now generate for the topic:"""

    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_text = message.choices[0].message.content.strip()

        # Try to parse as JSON
        try:
            qa_pair = json.loads(response_text)
            if "instruction" in qa_pair and "response" in qa_pair:
                qa_pair["topic"] = topic
                qa_pair["model"] = model
                return qa_pair
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Groq response as JSON: {response_text[:100]}")
            return None

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None


def generate_qa_pairs(
    topics: List[Dict[str, str]],
    max_pairs: int = DEFAULT_MAX_PAIRS,
    model: str = DEFAULT_MODEL,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
) -> List[Dict[str, Any]]:
    """Generate QA pairs from topics using Groq API with rate limiting.

    Args:
        topics: List of topic dicts with 'topic' key.
        max_pairs: Maximum number of pairs to generate.
        model: Groq model to use.
        rate_limit_delay: Delay in seconds between API calls.

    Returns:
        List of generated QA pair dicts.
    """
    client = get_groq_client()
    if not client:
        logger.error("Could not initialize Groq client. Check GROQ_API_KEY.")
        return []

    qa_pairs = []
    logger.info(f"Generating up to {max_pairs} QA pairs from {len(topics)} topics...")

    for i, topic_record in enumerate(topics[:max_pairs]):
        topic = topic_record.get("topic", "")
        if not topic:
            continue

        logger.info(f"[{i+1}/{min(max_pairs, len(topics))}] Generating for: {topic[:60]}...")

        qa_pair = generate_qa_pair(client, topic, model=model)
        if qa_pair:
            qa_pairs.append(qa_pair)
        else:
            logger.warning(f"Failed to generate QA pair for topic: {topic}")

        # Rate limiting: sleep between requests
        if i < min(max_pairs, len(topics)) - 1:
            time.sleep(rate_limit_delay)

    logger.info(f"Generated {len(qa_pairs)} QA pairs successfully")
    return qa_pairs


def run(
    topics_path: Path,
    output_path: Path,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    model: str = DEFAULT_MODEL,
) -> None:
    """Generate QA pairs from topics and save to JSONL.

    Args:
        topics_path: Path to topics JSONL file.
        output_path: Path to save raw QA pairs.
        max_pairs: Maximum pairs to generate (capped at 100 for free tier).
        model: Groq model to use.
    """
    # Cap at 100 for free tier
    max_pairs = min(max_pairs, 100)

    logger.info(f"Loading topics from {topics_path}...")
    topics = []
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            topics.append(json.loads(line))

    logger.info(f"Loaded {len(topics)} topics. Generating up to {max_pairs} QA pairs...")

    qa_pairs = generate_qa_pairs(topics, max_pairs=max_pairs, model=model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(qa_pairs)} QA pairs to {output_path}")


def main() -> None:
    """CLI entrypoint: generate QA pairs using Groq."""
    parser = argparse.ArgumentParser(description="Generate QA pairs using Groq LLM API.")
    parser.add_argument("--topics", type=Path, default=DATA_DIR / "synthetic" / "topics.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic" / "qa_pairs_raw.jsonl")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Max pairs to generate (capped at 100)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    run(
        topics_path=args.topics,
        output_path=args.output,
        max_pairs=min(args.max_pairs, 100),
        model=args.model,
    )


if __name__ == "__main__":
    main()
