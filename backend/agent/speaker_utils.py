
import json
import logging
from pathlib import Path
from typing import List, Any
from livekit.plugins.speechmatics import SpeakerIdentifier

logger = logging.getLogger("voice_agent")


def _truncate_identifier(identifier: str, max_length: int = 20) -> str:
    """Truncate a long speaker identifier for logging, showing first max_length chars followed by ..."""
    if len(identifier) > max_length:
        return f"{identifier[:max_length]}..."
    return identifier


def _truncate_identifier_list(identifiers: List[str]) -> List[str]:
    """Truncate a list of identifiers for logging"""
    return [_truncate_identifier(id) for id in identifiers]


def normalize_speaker_ids(raw_ids: Any) -> List[str]:
    """
    Normalize speaker IDs, supporting both old and new formats:
    - Old: ["id1", "id2"]
    - New: [{"label": "S1", "speaker_identifiers": ["id1"]}]

    Returns a list of strings, no duplicates, preserve order, no dicts, ignore invalid entries.
    """
    logger.debug(f"Raw speaker IDs type: {type(raw_ids)}")
    logger.debug(f"Raw speaker IDs value: {raw_ids!r}")

    normalized: List[str] = []
    seen = set()

    if not isinstance(raw_ids, list):
        logger.warning(f"Invalid raw speaker IDs: expected list, got {type(raw_ids)}")
        return []

    for idx, entry in enumerate(raw_ids):
        entry_log = f"Entry {idx}: {entry!r}"
        if isinstance(entry, str):
            if entry not in seen:
                seen.add(entry)
                normalized.append(entry)
            continue

        elif isinstance(entry, dict):
            speaker_identifiers = entry.get("speaker_identifiers")
            if isinstance(speaker_identifiers, list):
                for sub_id in speaker_identifiers:
                    if isinstance(sub_id, str) and sub_id not in seen:
                        seen.add(sub_id)
                        normalized.append(sub_id)
            else:
                logger.warning(
                    f"Skipping invalid dict entry: speaker_identifiers not a list of strings: {entry_log}"
                )

        else:
            logger.warning(f"Skipping invalid entry (not str or dict: {entry_log}")

    logger.debug(f"Normalized speaker IDs: {_truncate_identifier_list(normalized)}")
    logger.debug(f"Normalized speaker IDs count: {len(normalized)}")
    return normalized


def validate_speaker_identifiers(identifiers: Any) -> List[str]:
    """
    Validate and clean speaker_identifiers, ensuring it's a list of only strings.
    Logs warnings for invalid entries and removes them.
    Never crashes.
    """
    if not isinstance(identifiers, list):
        logger.warning(
            f"Invalid speaker_identifiers: expected list, got {type(identifiers)}. Returning empty list."
        )
        return []

    cleaned: List[str] = []
    for idx, identifier in enumerate(identifiers):
        if isinstance(identifier, str):
            cleaned.append(identifier)
        else:
            logger.warning(
                f"Skipping invalid speaker_identifier entry {idx}: not a string, got {type(identifier)} (value: {identifier!r})"
            )

    return cleaned


def load_known_speakers(file_path: Path) -> list[SpeakerIdentifier]:
    """Load known speakers from speakers.json, cleaning any invalid entries"""
    if not file_path.exists():
        logger.info("No existing speakers file found, creating empty one")
        save_speakers(file_path, [])
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            speakers_data = json.load(f)
            speakers = []
            for idx, sp in enumerate(speakers_data):
                try:
                    label = sp.get("label")
                    if not isinstance(label, str):
                        logger.warning(f"Skipping speaker {idx}: label not a string, got {type(label)} (value: {label!r})")
                        continue

                    # Clean speaker_identifiers
                    raw_speaker_ids = sp.get("speaker_identifiers", [])
                    cleaned_ids = validate_speaker_identifiers(raw_speaker_ids)
                    speakers.append(
                        SpeakerIdentifier(
                            label=label,
                            speaker_identifiers=cleaned_ids,
                        )
                    )

                except Exception as e:
                    logger.exception(f"Failed to process speaker {idx} ({sp!r}: {e}")

            logger.info(f"Loaded {len(speakers)} known speakers (cleaned from {len(speakers_data)} raw entries)")
            if speakers:
                logger.info("Known speaker labels:")
                for speaker in speakers:
                    logger.info(f"  - {speaker.label} (identifiers: {_truncate_identifier_list(speaker.speaker_identifiers)}")

            return speakers

    except Exception as e:
        logger.exception(f"Failed to load speakers from {file_path}: {e}")
        return []


def save_speakers(file_path: Path, speakers: list[SpeakerIdentifier]) -> None:
    """Save speakers to speakers.json, validating each speaker's identifiers first"""
    try:
        speakers_data = []
        for idx, sp in enumerate(speakers):
            try:
                cleaned_ids = validate_speaker_identifiers(sp.speaker_identifiers)
                speakers_data.append(
                    {
                        "label": sp.label,
                        "speaker_identifiers": cleaned_ids,
                    }
                )
            except Exception as e:
                logger.exception(f"Failed to process speaker {idx} for saving: {e}")
                continue

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(speakers_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(speakers_data)} speakers")
    except Exception as e:
        logger.exception(f"Failed to save speakers to {file_path}: {e}")
