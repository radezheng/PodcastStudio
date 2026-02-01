from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VibeVoiceConfig:
    model_id: str
    speaker_names: list[str]
    output_dir: Path
    vibevoice_repo_dir: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # PodcastStudio/


def _storage_root() -> Path:
    raw = os.environ.get("PODCASTSTUDIO_STORAGE_DIR", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        return p
    return (_project_root() / "storage").resolve()


def load_vibevoice_config() -> VibeVoiceConfig:
    model_id = os.environ.get("VIBEVOICE_MODEL_ID", "FabioSarracino/VibeVoice-Large-Q8").strip().strip('"')
    speaker_names_raw = os.environ.get("VIBEVOICE_SPEAKER_NAMES", "Xinran Anchen").strip().strip('"')
    output_dir = os.environ.get("VIBEVOICE_OUTPUT_DIR", "").strip().strip('"')

    speaker_names = [s for s in speaker_names_raw.split() if s]
    if not speaker_names:
        speaker_names = ["Xinran", "Anchen"]

    if output_dir:
        out = Path(output_dir)
    else:
        # default: shared storage (mountable to Azure Storage)
        out = _storage_root() / "outputs" / "tts_worker"

    # Use the bundled VibeVoice repo under this project.
    vibevoice_repo_dir = Path(__file__).resolve().parents[2] / "VibeVoice"

    return VibeVoiceConfig(
        model_id=model_id,
        speaker_names=speaker_names,
        output_dir=out,
        vibevoice_repo_dir=vibevoice_repo_dir,
    )


_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)$")


def normalize_to_speaker_script(text: str) -> str:
    """Ensure VibeVoice can parse the script.

    Accepted input formats:
    - Speaker 1: ... / Speaker 2: ... (preferred)
    - Any 'Name: ...' format (mapped to Speaker 1/2 by first-seen speakers)

    Output is always lines starting with 'Speaker N:'.
    """

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

        # Fallback: alternate speakers
        normalized.append(f"Speaker {next_id}: {ln}")
        next_id = 2 if next_id == 1 else 1

    return "\n".join(normalized) + "\n"


def run_vibevoice_inference(
    *,
    script_text: str,
    script_id: str | None = None,
    speaker_names: list[str] | None = None,
) -> Path:
    cfg = load_vibevoice_config()

    override_speakers = [s.strip() for s in (speaker_names or []) if s and s.strip()]
    if override_speakers and len(override_speakers) > 4:
        raise ValueError("speaker_names max is 4")

    speakers = override_speakers or cfg.speaker_names
    if not speakers:
        speakers = ["Xinran", "Anchen"]
    if len(speakers) > 4:
        speakers = speakers[:4]

    if not cfg.vibevoice_repo_dir.exists():
        raise RuntimeError(
            f"VibeVoice repo not found at {cfg.vibevoice_repo_dir}. "
            "Expected it under PodcastStudio/VibeVoice."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())
    safe_script_id = (script_id or "script").replace("/", "_")
    txt_path = cfg.output_dir / f"{safe_script_id}_{job_id}.txt"

    txt_path.write_text(normalize_to_speaker_script(script_text), encoding="utf-8")

    inference_py = cfg.vibevoice_repo_dir / "demo" / "inference_from_file.py"
    if not inference_py.exists():
        raise RuntimeError(f"Missing inference script: {inference_py}")

    # The demo uses relative outputs; we pass output_dir explicitly.
    cmd = [
        sys.executable,
        str(inference_py),
        "--model_path",
        cfg.model_id,
        "--txt_path",
        str(txt_path),
        "--output_dir",
        str(cfg.output_dir),
        "--speaker_names",
        *speakers,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(cfg.vibevoice_repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"VibeVoice inference failed (exit {proc.returncode}). Output:\n{proc.stdout}")

    # The demo prints "Saved output to <path>". Prefer parsing it.
    m = re.search(r"Saved output to\s+(?P<path>.+\.wav)", proc.stdout)
    if m:
        out_path = Path(m.group("path").strip())
        if not out_path.is_absolute():
            out_path = (cfg.vibevoice_repo_dir / out_path).resolve()
        if out_path.exists():
            return out_path

    # Fallback: newest wav in output_dir
    wavs = sorted(cfg.output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        return wavs[0]

    raise RuntimeError("VibeVoice finished but no wav file was found")
