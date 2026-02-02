from __future__ import annotations

import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .vibevoice_engine import get_engine
from .vibevoice_script import build_voice_samples, normalize_to_speaker_script, parse_speaker_script


@dataclass(frozen=True)
class VibeVoiceConfig:
    model_id: str
    speaker_names: list[str]
    output_dir: Path
    vibevoice_repo_dir: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # PodcastStudio/


def _resolve_vibevoice_repo_dir() -> Path:
    raw = os.environ.get("VIBEVOICE_REPO_DIR", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        return p

    candidates: list[Path] = [
        _project_root() / "VibeVoice",  # local dev / monorepo
        Path("/opt/VibeVoice"),  # tts_worker/Dockerfile clones here
        Path("/app/VibeVoice"),
    ]

    try:
        import vibevoice  # type: ignore

        candidates.append(Path(vibevoice.__file__).resolve().parents[1])
    except Exception:
        pass

    for candidate in candidates:
        if (candidate / "demo" / "inference_from_file.py").exists():
            return candidate

    return candidates[0]


def _storage_root() -> Path:
    raw = os.environ.get("PODCASTSTUDIO_STORAGE_DIR", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        return p
    return (_project_root() / "storage").resolve()


def load_vibevoice_config() -> VibeVoiceConfig:
    model_id = os.environ.get("VIBEVOICE_MODEL_ID", "vibevoice/VibeVoice-1.5B").strip().strip('"')
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

    vibevoice_repo_dir = _resolve_vibevoice_repo_dir()

    return VibeVoiceConfig(
        model_id=model_id,
        speaker_names=speaker_names,
        output_dir=out,
        vibevoice_repo_dir=vibevoice_repo_dir,
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _allow_subprocess_fallback() -> bool:
    # In production (Container Apps), we want deterministic behavior and avoid
    # the demo script's flash_attention_2/BF16 defaults, which can fail on T4.
    return _bool_env("VIBEVOICE_ALLOW_SUBPROCESS_FALLBACK", False)


def _dry_run_wav(path: Path) -> None:
    """Write a short, valid WAV file for smoke tests.

    Enabled by setting VIBEVOICE_DRY_RUN=1.
    """

    import math
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    duration_s = 0.25
    freq_hz = 440.0
    n = int(sample_rate * duration_s)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n):
            t = i / sample_rate
            v = int(0.2 * 32767.0 * math.sin(2.0 * math.pi * freq_hz * t))
            wf.writeframesraw(struct.pack("<h", v))


def run_vibevoice_inference(
    *,
    job_id: str,
    script_text: str,
    script_id: str | None = None,
    speaker_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    cfg = load_vibevoice_config()

    if output_dir is not None:
        cfg = VibeVoiceConfig(
            model_id=cfg.model_id,
            speaker_names=cfg.speaker_names,
            output_dir=output_dir,
            vibevoice_repo_dir=cfg.vibevoice_repo_dir,
        )

    if _bool_env("VIBEVOICE_DRY_RUN", False):
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        safe_script_id = (script_id or "script").replace("/", "_")
        wav_path = cfg.output_dir / f"{safe_script_id}_{job_id}_dryrun.wav"
        _dry_run_wav(wav_path)
        return wav_path

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
            "Set VIBEVOICE_REPO_DIR or ensure the repo is present (e.g. /opt/VibeVoice in the worker image)."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    safe_script_id = (script_id or "script").replace("/", "_")
    txt_path = cfg.output_dir / f"{safe_script_id}_{job_id}.txt"

    txt_path.write_text(normalize_to_speaker_script(script_text), encoding="utf-8")

    # Prefer in-process engine so the model stays loaded on GPU between jobs.
    # This drastically reduces per-job latency vs spawning a new Python process.
    if _bool_env("VIBEVOICE_INPROCESS", True):
        try:
            normalized = txt_path.read_text(encoding="utf-8")
            scripts, speaker_numbers = parse_speaker_script(normalized)
            full_script = "\n".join(scripts).replace("’", "'")

            voice_samples = build_voice_samples(
                vibevoice_repo_dir=cfg.vibevoice_repo_dir,
                scripts=scripts,
                speaker_numbers=speaker_numbers,
                speaker_names=speakers,
            )

            wav_path = cfg.output_dir / f"{safe_script_id}_{job_id}_generated.wav"
            get_engine().synthesize(text=full_script, voice_samples=voice_samples, output_path=wav_path)
            if wav_path.exists():
                return wav_path
        except Exception as e:
            # Optional fallback: useful for local debugging, but disabled by default.
            if not _allow_subprocess_fallback():
                raise
            last_error = e
    else:
        last_error = None

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

    import subprocess

    proc = subprocess.run(
        cmd,
        cwd=str(cfg.vibevoice_repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        extra = ""
        try:
            if "last_error" in locals() and last_error is not None:
                extra = f"\n(In-process attempt failed first: {type(last_error).__name__}: {last_error})"
        except Exception:
            pass
        raise RuntimeError(f"VibeVoice inference failed (exit {proc.returncode}). Output:\n{proc.stdout}{extra}")

    # The demo prints "Saved output to <path>". Prefer parsing it.
    m = re.search(r"Saved output to\s+(?P<path>.+\.wav)", proc.stdout)
    if m:
        out_path = Path(m.group("path").strip())
        if not out_path.is_absolute():
            out_path = (cfg.vibevoice_repo_dir / out_path).resolve()
        if out_path.exists():
            desired = cfg.output_dir / f"{safe_script_id}_{job_id}_generated.wav"
            if out_path != desired:
                try:
                    desired.parent.mkdir(parents=True, exist_ok=True)
                    desired.write_bytes(out_path.read_bytes())
                    return desired
                except Exception:
                    return out_path
            return out_path

    # Fallback: newest wav in output_dir
    wavs = sorted(cfg.output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        # Prefer returning a stable name inside output_dir if possible.
        desired = cfg.output_dir / f"{safe_script_id}_{job_id}_generated.wav"
        if wavs[0] != desired:
            try:
                desired.parent.mkdir(parents=True, exist_ok=True)
                desired.write_bytes(wavs[0].read_bytes())
                return desired
            except Exception:
                return wavs[0]
        return wavs[0]

    raise RuntimeError("VibeVoice finished but no wav file was found")
