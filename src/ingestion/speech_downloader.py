"""Downloader for OpenSLR SLR65 (Tamil TTS corpus) and Common Voice Tamil.

Sources:
    - OpenSLR SLR65: "Crowdsourced high-quality Tamil multi-speaker speech
      dataset" (Google, CC-BY-SA 4.0). Served as two large zips
      (``ta_in_male.zip`` / ``ta_in_female.zip``) directly over HTTP with no
      authentication. Because these zips are large (500MB-750MB) and the
      OpenSLR server supports HTTP range requests, we use ``fsspec``'s HTTP
      filesystem to open the remote zip and read *only* the audio entries we
      need (per the ``--limit``/``--max-samples`` cap), without downloading
      the whole archive. Pass ``--limit -1`` (or ``--max-samples -1``) to
      download every entry (for a full production run).
    - Common Voice Tamil: pulled via the Hugging Face
      ``mozilla-foundation/common_voice_17_0`` dataset (config ``ta``) in
      streaming mode. This dataset is access-gated (free, click-through
      license agreement): create a free HF account, accept the terms at
      https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0,
      generate a token at https://huggingface.co/settings/tokens, and put it
      in ``.env`` as ``HF_TOKEN``. Without a valid token this source is
      skipped with a warning so the rest of the pipeline can still run.

Run standalone:
    python -m src.ingestion.speech_downloader --limit 200
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

from src.common.io_utils import write_parquet
from src.common.paths import AUDIO_DIR, RAW_DIR
from src.common.schema import make_speech_record
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENSLR_BASE_URL = "https://www.openslr.org/resources/65"
OPENSLR_ZIPS = {
    "male": f"{OPENSLR_BASE_URL}/ta_in_male.zip",
    "female": f"{OPENSLR_BASE_URL}/ta_in_female.zip",
}
OPENSLR_LICENSE = "CC-BY-SA-4.0"

COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_17_0"
COMMON_VOICE_CONFIG = "ta"
COMMON_VOICE_LICENSE = "CC0-1.0"


def _parse_line_index(raw: bytes) -> Dict[str, str]:
    """Parse an OpenSLR ``line_index.tsv`` blob into a ``{file_id: transcript}`` dict."""
    mapping: Dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        file_id, transcript = parts
        mapping[file_id.strip()] = transcript.strip()
    return mapping


def download_openslr(gender: str, zip_url: str, max_samples: Optional[int], audio_dir: Path) -> List[Dict[str, Any]]:
    """Download (or partially read) one OpenSLR SLR65 gender-split zip archive.

    Uses ``fsspec``'s HTTP filesystem to open the remote zip with random
    access (via HTTP range requests), so only the entries we actually need
    are transferred over the network.

    Args:
        gender: ``"male"`` or ``"female"`` — used for speaker metadata and file naming.
        zip_url: Direct HTTP URL of the OpenSLR zip archive.
        max_samples: Max number of audio clips to extract (``None`` = every entry).
        audio_dir: Local directory to save extracted ``.wav`` files under.

    Returns:
        List of schema-compliant speech records.
    """
    import fsspec

    logger.info("Opening %s (gender=%s) via HTTP range requests ...", zip_url, gender)
    fs = fsspec.filesystem("http")
    with fs.open(zip_url, mode="rb") as remote_file:
        zf = zipfile.ZipFile(remote_file)
        names = zf.namelist()

        line_index: Dict[str, str] = {}
        if "line_index.tsv" in names:
            line_index = _parse_line_index(zf.read("line_index.tsv"))

        wav_names = [n for n in names if n.endswith(".wav")]
        if max_samples is not None:
            wav_names = wav_names[:max_samples]

        out_dir = audio_dir / "openslr_slr65" / gender
        out_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for name in wav_names:
            file_id = Path(name).stem
            transcript = line_index.get(file_id, "")
            dest = out_dir / Path(name).name
            if not dest.exists():
                dest.write_bytes(zf.read(name))
            records.append(
                make_speech_record(
                    audio_path=str(dest.relative_to(audio_dir.parent.parent)),
                    source="openslr:slr65",
                    license=OPENSLR_LICENSE,
                    pipeline_stage="raw",
                    transcript=transcript,
                    speaker_gender=gender,
                    file_id=file_id,
                )
            )
        logger.info("Extracted %d clips for gender=%s", len(records), gender)
        return records


def download_common_voice(max_samples: Optional[int], audio_dir: Path, hf_token: Optional[str]) -> List[Dict[str, Any]]:
    """Stream Common Voice Tamil clips from the Hugging Face Hub and save them locally.

    Args:
        max_samples: Max number of clips to pull (``None`` = unlimited, not recommended).
        audio_dir: Local directory to save extracted audio files under.
        hf_token: HF access token; required since this dataset needs a click-through
            license agreement even though it is free.

    Returns:
        List of schema-compliant speech records. Empty if the dataset could not be accessed.
    """
    if not hf_token:
        logger.warning(
            "No HF_TOKEN found in .env — skipping Common Voice Tamil download. "
            "Accept the license at https://huggingface.co/datasets/%s and set HF_TOKEN to enable this source.",
            COMMON_VOICE_DATASET,
        )
        return []

    from datasets import Audio, load_dataset

    out_dir = audio_dir / "common_voice_ta"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(COMMON_VOICE_DATASET, COMMON_VOICE_CONFIG, split="train", streaming=True, token=hf_token)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:  # noqa: BLE001 - keep pipeline resilient to gating/network issues
        logger.warning("Failed to open Common Voice Tamil (%s): %s", COMMON_VOICE_DATASET, exc)
        return []

    records = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        audio = row.get("audio") or {}
        raw_bytes = audio.get("bytes")
        src_path = audio.get("path") or f"cv_{i}.mp3"
        dest = out_dir / f"cv_{i:06d}_{Path(src_path).name}"
        if raw_bytes:
            dest.write_bytes(raw_bytes)
        records.append(
            make_speech_record(
                audio_path=str(dest.relative_to(audio_dir.parent.parent)),
                source=f"hf:{COMMON_VOICE_DATASET}:{COMMON_VOICE_CONFIG}",
                license=COMMON_VOICE_LICENSE,
                pipeline_stage="raw",
                transcript=row.get("sentence"),
                speaker_id=row.get("client_id"),
                speaker_gender=row.get("gender"),
                speaker_age=row.get("age"),
                accent=row.get("accent"),
            )
        )
    logger.info("Pulled %d Common Voice Tamil clips.", len(records))
    return records


def run(
    output_path: Path,
    openslr_max_per_gender: Optional[int],
    common_voice_max: Optional[int],
    audio_dir: Path,
    hf_token: Optional[str],
    include_common_voice: bool = True,
) -> None:
    """Download speech sources and write a combined manifest Parquet file.

    Args:
        output_path: Destination ``.parquet`` file for the speech manifest.
        openslr_max_per_gender: Max clips per OpenSLR gender split (``None`` = unlimited).
        common_voice_max: Max clips to pull from Common Voice (``None`` = unlimited).
        audio_dir: Local directory to save audio files under.
        hf_token: HF access token for Common Voice (gated dataset).
        include_common_voice: Whether to attempt the Common Voice download.
    """
    all_records: List[Dict[str, Any]] = []

    for gender, url in OPENSLR_ZIPS.items():
        try:
            all_records.extend(download_openslr(gender, url, openslr_max_per_gender, audio_dir))
        except Exception as exc:  # noqa: BLE001 - keep pipeline resilient to one bad source
            logger.warning("Failed to download OpenSLR SLR65 (%s): %s", gender, exc)

    cv_records: List[Dict[str, Any]] = []
    if include_common_voice:
        cv_records = download_common_voice(common_voice_max, audio_dir, hf_token)
        all_records.extend(cv_records)

    write_parquet(all_records, output_path)
    logger.info("Wrote %d speech records to %s", len(all_records), output_path)

    log_stage_stats(
        stage="ingestion.speech_downloader",
        input_count=len(all_records),
        output_count=len(all_records),
        removed_count=0,
        extra={
            "openslr_max_per_gender": openslr_max_per_gender,
            "common_voice_max": common_voice_max,
            "openslr_count": len(all_records) - len(cv_records),
            "common_voice_count": len(cv_records),
        },
    )


def main() -> None:
    """CLI entrypoint: download speech sources and write the raw speech manifest."""
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description="Download OpenSLR SLR65 + Common Voice Tamil speech data.")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "speech_manifest.parquet", help="Output manifest Parquet path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Total clip cap combined across Common Voice + OpenSLR (demo/local-sized default; use -1 for unlimited).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Advanced override: exact cap applied to each of OpenSLR-male, OpenSLR-female, and Common Voice "
        "independently, bypassing --limit's split (use -1 for unlimited/full download).",
    )
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR, help="Directory to save audio files under.")
    parser.add_argument("--skip-common-voice", action="store_true", help="Skip the Common Voice Tamil source.")
    args = parser.parse_args()

    if args.max_samples is not None:
        # Advanced path: same cap applied independently to each of the 3 buckets (male/female/common-voice).
        max_samples = None if args.max_samples < 0 else args.max_samples
        openslr_max_per_gender = common_voice_max = max_samples
    elif args.limit is not None and args.limit >= 0:
        # Default path: split the total combined --limit ~50/50 between OpenSLR and Common Voice,
        # then split the OpenSLR half ~50/50 between the male/female zips.
        common_voice_max = args.limit // 2
        openslr_total = args.limit - common_voice_max
        openslr_max_per_gender = max(1, openslr_total // 2)
    else:
        openslr_max_per_gender = common_voice_max = None

    hf_token = os.getenv("HF_TOKEN") or None

    run(
        output_path=args.output,
        openslr_max_per_gender=openslr_max_per_gender,
        common_voice_max=common_voice_max,
        audio_dir=args.audio_dir,
        hf_token=hf_token,
        include_common_voice=not args.skip_common_voice,
    )

    # See hf_loader.py: `datasets`/`huggingface_hub` background threads can
    # otherwise hang process exit even though all work is already on disk.
    os._exit(0)


if __name__ == "__main__":
    main()
