from __future__ import annotations

import os
import re
from pathlib import Path


_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)


def normalize_to_speaker_script(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    if not lines:
        raise ValueError("script_text is empty")

    if all(_SPEAKER_LINE.match(ln) for ln in lines):
        return "\n".join(lines) + "\n"

    speaker_map: dict[str, int] = {}
    next_id = 1
    normalized: list[str] = []

    for ln in lines:
        if _SPEAKER_LINE.match(ln):
            normalized.append(ln)
            continue

        if ":" in ln:
            name, rest = ln.split(":", 1)
            name = name.strip()
            rest = rest.strip()
            if name and rest:
                if name not in speaker_map:
                    speaker_map[name] = next_id
                    next_id = 2 if next_id == 1 else 1
                normalized.append(f"Speaker {speaker_map[name]}: {rest}")
                continue

        normalized.append(f"Speaker {next_id}: {ln}")
        next_id = 2 if next_id == 1 else 1

    return "\n".join(normalized) + "\n"


def parse_speaker_script(text: str) -> tuple[list[str], list[str]]:
    """Return (segments, speaker_numbers) for normalized 'Speaker N:' scripts."""
    lines = (text or "").strip().split("\n")
    scripts: list[str] = []
    speaker_numbers: list[str] = []

    current_speaker: str | None = None
    current_text = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            current_speaker = m.group(1).strip()
            current_text = m.group(2).strip()
        else:
            current_text = (current_text + " " + line).strip() if current_text else line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


class VoiceMapper:
    def __init__(self, *, voices_dir: Path):
        self.voices_dir = voices_dir
        self.voice_presets = self._scan()

    def _scan(self) -> dict[str, str]:
        if not self.voices_dir.exists():
            return {}

        presets: dict[str, str] = {}
        for wav in self.voices_dir.glob("*.wav"):
            name = wav.stem
            # match upstream demo normalization
            if "_" in name:
                name = name.split("_", 1)[0]
            if "-" in name:
                name = name.split("-")[-1]
            name = name.strip()
            if name:
                presets[name] = str(wav)
        return presets

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            pn = preset_name.lower()
            if pn in speaker_lower or speaker_lower in pn:
                return path

        if not self.voice_presets:
            raise RuntimeError(
                f"No voice presets found under {self.voices_dir}. "
                "Ensure VibeVoice demo voices are present in the image."
            )

        return next(iter(self.voice_presets.values()))


def _vibevoice_voices_dir(vibevoice_repo_dir: Path) -> Path:
    # In upstream repo: demo/voices
    return vibevoice_repo_dir / "demo" / "voices"


def build_voice_samples(
    *,
    vibevoice_repo_dir: Path,
    scripts: list[str],
    speaker_numbers: list[str],
    speaker_names: list[str],
) -> list[str]:
    # Map Speaker 1..4 -> provided names (by order)
    mapping: dict[str, str] = {}
    for i, name in enumerate(speaker_names, 1):
        mapping[str(i)] = name

    unique_speaker_numbers: list[str] = []
    seen: set[str] = set()
    for num in speaker_numbers:
        if num not in seen:
            unique_speaker_numbers.append(num)
            seen.add(num)

    mapper = VoiceMapper(voices_dir=_vibevoice_voices_dir(vibevoice_repo_dir))

    voice_samples: list[str] = []
    for speaker_num in unique_speaker_numbers:
        speaker_name = mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_samples.append(mapper.get_voice_path(speaker_name))

    return voice_samples
