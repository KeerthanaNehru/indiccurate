"""QA pair validation using LLM-as-judge (second Groq call).

Each generated QA pair is scored on:
- Coherence: Is the response coherent and grammatically correct?
- Relevance: Does the response address the instruction?
- Factuality: Is the response factually sound?
- Quality: Overall quality on a 1-5 scale.

Pairs scoring below a threshold are discarded.

Run standalone:
    python -m src.synthetic.qa_validator --input data/synthetic/qa_pairs_raw.jsonl \
        --output data/synthetic/qa_pairs_validated.jsonl --threshold 3
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

DEFAULT_THRESHOLD = 3.0  # Min score out of 5
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_RATE_LIMIT_DELAY = 0.5


def get_groq_client():
    """Initialize Groq client from GROQ_API_KEY environment variable."""
    try:
        from groq import Groq
        import os

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set in environment.")
            return None

        return Groq(api_key=api_key)
    except (ImportError, Exception) as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None


def score_qa_pair(
    client,
    instruction: str,
    response: str,
    model: str = DEFAULT_MODEL,
) -> Optional[Dict[str, Any]]:
    """Score a QA pair using LLM-as-judge.

    Args:
        client: Groq client instance.
        instruction: The instruction/question.
        response: The generated response.
        model: Groq model to use.

    Returns:
        Dict with scores and feedback, or None on error.
    """
    prompt = f"""You are an expert evaluator of Tamil language content. 
Evaluate the following instruction-response pair on quality dimensions.

Instruction (Tamil): {instruction}

Response (Tamil): {response}

Rate the pair on these dimensions (1-5 scale, where 5 is excellent):
1. Coherence: Is the response coherent, grammatically correct, and well-structured?
2. Relevance: Does the response directly address the instruction?
3. Factuality: Is the information factually accurate and trustworthy?
4. Overall Quality: Overall assessment of the QA pair quality.

Provide your evaluation as JSON with keys: coherence, relevance, factuality, overall_quality.
Each value should be an integer from 1 to 5.
Include a 'feedback' key with a brief explanation.

Example:
{{"coherence": 4, "relevance": 5, "factuality": 4, "overall_quality": 4, "feedback": "Good quality pair with minor issues."}}

Now evaluate:"""

    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_text = message.choices[0].message.content.strip()

        try:
            scores = json.loads(response_text)
            # Ensure required keys exist
            if all(k in scores for k in ["coherence", "relevance", "factuality", "overall_quality"]):
                return scores
        except json.JSONDecodeError:
            logger.warning(f"Could not parse judge response as JSON: {response_text[:100]}")
            return None

    except Exception as e:
        logger.error(f"Groq API error during scoring: {e}")
        return None


def validate_qa_pairs(
    qa_pairs: List[Dict[str, str]],
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
) -> tuple[List[Dict[str, Any]], int]:
    """Validate QA pairs using LLM-as-judge.

    Args:
        qa_pairs: List of QA pair dicts.
        threshold: Minimum overall_quality score to keep (1-5).
        model: Groq model to use.
        rate_limit_delay: Delay between API calls.

    Returns:
        Tuple of (validated_pairs, rejected_count).
    """
    client = get_groq_client()
    if not client:
        logger.error("Could not initialize Groq client.")
        return [], len(qa_pairs)

    validated = []
    rejected = 0

    logger.info(f"Validating {len(qa_pairs)} QA pairs (threshold: {threshold})...")

    for i, qa_pair in enumerate(qa_pairs):
        instruction = qa_pair.get("instruction", "")
        response = qa_pair.get("response", "")

        if not instruction or not response:
            logger.warning(f"Skipping pair {i+1}: missing instruction or response")
            rejected += 1
            continue

        logger.info(f"[{i+1}/{len(qa_pairs)}] Validating: {instruction[:50]}...")

        scores = score_qa_pair(client, instruction, response, model=model)

        if scores:
            overall_quality = scores.get("overall_quality", 0)
            if overall_quality >= threshold:
                # Add scores to the pair
                qa_pair["scores"] = scores
                qa_pair["validated"] = True
                validated.append(qa_pair)
                logger.info(f"  ✓ Accepted (quality: {overall_quality}/5)")
            else:
                logger.info(f"  ✗ Rejected (quality: {overall_quality}/5 < {threshold})")
                rejected += 1
        else:
            logger.warning(f"Failed to score pair {i+1}")
            rejected += 1

        # Rate limiting
        if i < len(qa_pairs) - 1:
            time.sleep(rate_limit_delay)

    logger.info(f"Validation complete: {len(validated)} kept, {rejected} rejected")
    return validated, rejected


def run(
    input_path: Path,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
) -> None:
    """Validate QA pairs and save validated ones.

    Args:
        input_path: Path to raw QA pairs JSONL.
        output_path: Path to save validated QA pairs.
        threshold: Min score to keep (1-5).
        model: Groq model to use.
    """
    logger.info(f"Loading QA pairs from {input_path}...")
    qa_pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    logger.info(f"Loaded {len(qa_pairs)} QA pairs. Running validation...")

    validated, rejected = validate_qa_pairs(qa_pairs, threshold=threshold, model=model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in validated:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(validated)} validated QA pairs to {output_path}")


def main() -> None:
    """CLI entrypoint: validate QA pairs using LLM-as-judge."""
    parser = argparse.ArgumentParser(description="Validate QA pairs using LLM-as-judge scoring.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "synthetic" / "qa_pairs_raw.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic" / "qa_pairs_validated.jsonl")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Min quality score (1-5)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        model=args.model,
    )


if __name__ == "__main__":
    main()
